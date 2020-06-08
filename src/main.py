import concurrent.futures
import csv
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta

import logging_gelf.handlers
import logging_gelf.formatters

import numpy as np
import pandas as pd
import pytz
import requests
from keboola import docker

timeout_delta = timedelta(minutes=50)
EXECUTION_START = datetime.now()
URL_BASE = 'https://api.zbozi.cz'
BATCH_SIZE_PRODUCTS = 10
BATCH_SIZE_SHOPS = 100
CURRENT_TIME = datetime.now(pytz.timezone('Europe/Prague'))
TS = CURRENT_TIME.strftime('%Y%m%d%H%M%S')
COUNTRY = 'CZ'
DISTRCHAN = 'MA'
SOURCE = 'zbozi'
FREQ = 'd'
SOURCE_ID = f'{SOURCE}_{TS}'
CURRENT_DATE_STR = datetime.now().strftime('%Y-%m-%d')


class DailyScrapingFinishedError(Exception):
    pass


class Producer(object):
    def __init__(self, datadir, pipeline):
        self.task_queue = pipeline
        self.datadir = datadir
        params = docker.Config(self.datadir).get_parameters()
        self.login = params.get('#login')
        self.password = params.get('#password')
        self.out_cols = ['AVAILABILITY', 'COUNTRY', 'CSE_ID', 'CSE_URL', 'DISTRCHAN', 'ESHOP', 'FREQ',
                         'HIGHLIGHTED_POSITION', 'MATERIAL', 'POSITION', 'PRICE', 'RATING',
                         'REVIEW_COUNT', 'SOURCE', 'SOURCE_ID', 'STOCK', 'TOP', 'TS', 'URL']
        additional_cols = ['ZBOZI_SHOP_ID', 'MATCHING_ID', 'DATE']
        self.all_cols = self.out_cols + additional_cols
        self.export_table = 'results'

    @property
    def auth(self):
        return self.login, self.password

    def update_keep_scraping(self, value):
        pd.DataFrame({'KS': [value]}).to_csv('../data/out/tables/keep_scraping.csv', index=False)

    def gen_seq_chunks(self, seq, chunk_size):
        for pos in range(0, len(seq), chunk_size):
            yield seq[pos:pos + chunk_size]

    def get_products(self, ids_str):
        response = requests.get(f'{URL_BASE}/v1/products/{ids_str}', auth=self.auth)
        if response.status_code // 100 != 2:
            logging.warning(f'failed products get request: {response.status_code, response.text}')
            return None

        content = json.loads(response.text)
        product_pages = pd.DataFrame()
        try:
            for data in content['data']:
                if not data.get('shopItems') and not data.get('topShopItems'):
                    # skip invalid shops
                    continue

                # cast all values to str so pandas will cast it to object arrays
                shops_df = pd.DataFrame({k: str(v) if v is not None else '' for
                                         k, v in d.items()} for d in data.get('shopItems'))
                shops_df['POSITION'] = np.arange(1, len(shops_df) + 1)

                h_shops_df = pd.DataFrame({k: str(v) if v is not None else '' for
                                           k, v in d.items()} for d in data.get('topShopItems'))
                h_shops_df['HIGHLIGHTED_POSITION'] = np.arange(1, len(h_shops_df) + 1)

                product_page = pd.concat([shops_df, h_shops_df], sort=True)
                product_page['DATE'] = datetime.now().strftime('%Y-%m-%d')
                product_page['TS'] = datetime.now().strftime('%Y%m%d%H%M%S')
                product_page['CSE_ID'] = str(data['productId'])
                product_page = product_page.groupby('shopId', as_index=False).first()
                product_pages = pd.concat([product_pages, product_page], sort=True)

        except Exception as e:
            logging.warning(f'Exception: {e}, data: {str(content)}')
            return None
        else:
            return product_pages

    def get_shop_names(self, shop_ids_str):
        response = requests.get(f'{URL_BASE}/v1/shops/{shop_ids_str}',
                                auth=self.auth)

        if response.status_code // 100 != 2:
            return None
        return [{k: str(v) if v is not None else '' for k, v in d.items()} for d in response.json()['data']]

    def save_out_subtables(self, products_df):
        products_df.reset_index(drop=True, inplace=True)
        to_send = products_df[self.out_cols].fillna('').to_dict('records')
        self.task_queue.put(to_send)

    def save_out_tables(self, products_df, shop_names=None, material_map=None):
        if shop_names is not None:
            shops_df = pd.DataFrame(shop_names, columns=['name', 'rating', 'shopId'])
            shops_df.rename(columns={'name': 'ESHOP',
                                     'rating': 'RATING',
                                     'shopId': 'ZBOZI_SHOP_ID',
                                     'availability': 'AVAILABILITY'},
                            inplace=True)
            eshop_notnull_index = pd.notnull(products_df['ESHOP'])
            products_df_w_shop_info = products_df[eshop_notnull_index]
            products_df_wo_shop_info = products_df[~eshop_notnull_index]

            products_df_wo_shop_info = products_df_wo_shop_info.drop(['ESHOP', 'RATING'], axis=1)
            products_df = pd.merge(products_df_wo_shop_info, shops_df,
                                   on='ZBOZI_SHOP_ID', how='left')
            products_df = pd.concat([products_df, products_df_w_shop_info], sort=True)

            products_df['ESHOP'] = products_df['ESHOP'].str.lower()

        if material_map is not None:
            mat_notnull_index = pd.notnull(products_df['MATERIAL'])
            products_df_w_mat_info = products_df[mat_notnull_index]
            products_df_wo_mat_info = products_df[~mat_notnull_index]

            products_df_wo_mat_info = products_df_wo_mat_info.drop('MATERIAL', axis=1)
            products_df = pd.merge(products_df_wo_mat_info, material_map,
                                   on='CSE_ID', how='left')
            products_df = pd.concat([products_df, products_df_w_mat_info], sort=True)

        products_df = products_df.groupby(['DATE', 'CSE_ID', 'ZBOZI_SHOP_ID'],
                                          as_index=False).first()

        products_df['TS'] = TS  # rewrite timestamp from previous run
        products_df['COUNTRY'] = COUNTRY
        products_df['DISTRCHAN'] = DISTRCHAN
        products_df['SOURCE'] = SOURCE
        products_df['SOURCE_ID'] = SOURCE_ID
        products_df['FREQ'] = FREQ
        products_df.loc[:, 'STOCK'] = np.where(products_df['AVAILABILITY'] == 0, '1', '0')

        products_df.to_csv('../data/out/tables/ZBOZI_DAILY.csv',
                           index=False, columns=self.all_cols)
        return products_df

    def produce(self):
        try:
            self._produce()
        except Exception as e:
            logging.exception(f'Error occurred {e}')
        finally:
            # send ending token
            self.task_queue.put('DONE')

    def _produce(self):
        keep_scraping = pd.read_csv('../data/in/tables/keep_scraping.csv')
        if not keep_scraping.iloc[0, 0]:
            logging.error(f'Keep scraping: {keep_scraping.iloc[0, 0]}')
            raise DailyScrapingFinishedError

        try:
            products_df = pd.read_csv('../data/in/tables/ZBOZI_DAILY.csv')
            products_df = products_df[products_df['DATE'] == CURRENT_DATE_STR]
        except Exception as e:
            logging.warning(f'Loading partial data failed with: {e}')
            products_df = pd.DataFrame(columns=self.all_cols)

        skip_execution = False
        ##########################################################################
        # PRODUCT IDS
        all_product_ids = []
        next_url = '/v1/shop/items?paired=True&limit=1000&loadProductDetail=False'

        logging.info('Start product requests')
        # counter = 0
        while next_url is not None:
            response = requests.get(f'{URL_BASE}{next_url}', auth=self.auth)
            if response.status_code // 100 != 2:
                time.sleep(1.01)
                continue

            content = json.loads(response.text)
            product_ids = [(str(item['itemId']), str(item['product']['productId']))
                           for item in content.get('data', list())
                           if item.get('product') is not None
                           ]
            all_product_ids.extend(product_ids)
            next_url = content.get('links', dict()).get('next')
            # counter += 1
            # if counter >= 5:
            #     break

        logging.info('End product requests')
        material_map = pd.DataFrame(all_product_ids, columns=['MATERIAL', 'CSE_ID']).astype(object)
        all_pids = material_map['CSE_ID'].to_list()

        new_product_ids = list(set(all_pids).difference(products_df['CSE_ID'].to_list()))

        #############################################################################
        # PRODUCT PAGES
        failed_product_ids_strs = list()

        for ids_group in self.gen_seq_chunks(new_product_ids, BATCH_SIZE_PRODUCTS):
            ids_str = ','.join(map(str, map(int, ids_group)))
            failed_product_ids_strs.append(ids_str)

        new_products_df = pd.DataFrame()
        logging.info('Start product pages requests')
        while failed_product_ids_strs:
            ids_strs = failed_product_ids_strs[:]
            failed_product_ids_strs = list()

            for ids_str in ids_strs:
                # counter += 1
                # if counter >= 20:
                #     break

                product_batch_df = self.get_products(ids_str)
                if product_batch_df is not None:
                    new_products_df = pd.concat([new_products_df, product_batch_df], sort=True)
                    time.sleep(0.23)
                else:
                    logging.info(f'product_ids_str: {ids_str} failed')
                    failed_product_ids_strs.append(ids_str)
                    time.sleep(1.21)

                skip_execution = (datetime.now() - EXECUTION_START) >= timeout_delta
                if skip_execution:
                    new_products_df = new_products_df.rename(
                        columns={'shopId': 'ZBOZI_SHOP_ID',
                                 'availability': 'AVAILABILITY',
                                 'matchingId': 'MATCHING_ID',
                                 'price': 'PRICE'})
                    products_df = pd.concat([products_df, new_products_df], sort=True)
                    products_df = self.save_out_tables(products_df, material_map=material_map)
                    self.save_out_subtables(products_df)
                    self.update_keep_scraping(value=1)
                    break
            if skip_execution:
                break

        logging.info('End product pages requests')

        ###############################################################################
        # SHOP NAMES
        logging.info('Start shops requests')
        all_shop_names = list()
        if not skip_execution:
            new_products_df.rename(columns={'shopId': 'ZBOZI_SHOP_ID',
                                            'matchingId': 'MATCHING_ID',
                                            'price': 'PRICE',
                                            'availability': 'AVAILABILITY'},
                                   inplace=True)
            products_df = pd.concat([products_df, new_products_df], sort=True)
            remaining_shop_ids = products_df.loc[(products_df['ESHOP'].isnull()) &
                                                 (pd.notnull(products_df['ZBOZI_SHOP_ID'])),
                                                 'ZBOZI_SHOP_ID'].astype(int).astype(str).unique().tolist()

            failed_shop_ids_strs = [','.join(map(str, map(int, shop_ids_group)))
                                    for shop_ids_group in self.gen_seq_chunks(remaining_shop_ids, BATCH_SIZE_SHOPS)
                                    ]

            while failed_shop_ids_strs:
                shop_ids_strs = failed_shop_ids_strs[:]
                failed_shop_ids_strs = list()

                for shop_ids_str in shop_ids_strs:
                    shop_names = self.get_shop_names(shop_ids_str)
                    if shop_names is not None:
                        all_shop_names.extend(shop_names)
                        time.sleep(1.01)
                    else:
                        failed_shop_ids_strs.append(shop_ids_str)
                        time.sleep(1)

                    skip_execution = (datetime.now() - EXECUTION_START) >= timeout_delta
                    if skip_execution:
                        products_df = self.save_out_tables(products_df, shop_names=all_shop_names,
                                                           material_map=material_map)
                        self.save_out_subtables(products_df)
                        self.update_keep_scraping(value=1)
                        break
                if skip_execution:
                    break

        logging.info('End shops requests')
        skip_execution = (datetime.now() - EXECUTION_START) >= timeout_delta
        if not skip_execution:
            logging.info('Start writeout')
            products_df = self.save_out_tables(products_df, shop_names=all_shop_names, material_map=material_map)
            self.save_out_subtables(products_df)
            self.update_keep_scraping(value=0)
            logging.info('End writeout')


class Writer(object):
    def __init__(self, task_queue, columns_list, threading_event, filepath):
        self.task_queue = task_queue
        self.columns_list = columns_list
        self.threading_event = threading_event
        self.filepath = filepath

    def write(self):
        with open(self.filepath, 'w+') as outfile:
            results_writer = csv.DictWriter(outfile, fieldnames=self.columns_list, extrasaction='ignore')
            results_writer.writeheader()
            while not self.threading_event.is_set():
                chunk = self.task_queue.get()
                if chunk == 'DONE':
                    logging.info('DONE received. Exiting.')
                    self.threading_event.set()
                else:
                    results_writer.writerows(chunk)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, handlers=[])  # do not create default stdout handler
    logger = logging.getLogger()
    logging_gelf_handler = logging_gelf.handlers.GELFTCPSocketHandler(
        host=os.getenv('KBC_LOGGER_ADDR'),
        port=int(os.getenv('KBC_LOGGER_PORT')))
    logging_gelf_handler.setFormatter(logging_gelf.formatters.GELFFormatter(null_character=True))
    logger.addHandler(logging_gelf_handler)

    logging.info(f'Starting run of "{SOURCE_ID}"')
    colnames = ['AVAILABILITY',
                'COUNTRY',
                'CSE_ID',
                'CSE_URL',
                'DISTRCHAN',
                'ESHOP',
                'FREQ',
                'HIGHLIGHTED_POSITION',
                'MATERIAL',
                'POSITION',
                'PRICE',
                'RATING',
                'REVIEW_COUNT',
                'SOURCE',
                'SOURCE_ID',
                'STOCK',
                'TOP',
                'TS',
                'URL']

    datadir = os.getenv('KBC_DATADIR', '/data/')
    path = f'{os.getenv("KBC_DATADIR")}out/tables/results.csv'
    pipeline = queue.Queue(maxsize=1000)
    event = threading.Event()
    producer = Producer(datadir, pipeline)
    writer = Writer(pipeline, colnames, event, path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer.produce)
        executor.submit(writer.write)
