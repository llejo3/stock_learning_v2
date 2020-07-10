import os
import time
from datetime import date

import pandas as pd
import requests
from tqdm import tqdm

import logging_config as log
from corp_loader import CorpLoader
from utils.data_utils import DataUtils
from utils.date_utils import DateUtils


class StockLoader:
    """
    주식 데이터를 불러온다.
    """

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    STOCK_DIR = os.path.join(ROOT_DIR, 'data', 'stocks')
    WEB_SEARCH_FIRST = 'daum'

    def __init__(self):
        self.logger = log.get_logger(self.__class__.__name__)

    # @DataUtils.clock
    def get_stock_data(self, corp_code: str, cnt_to_del=0, update_stock=True):
        """
        주식 데이터를 가져온다.
        :param corp_code: 주식회사 코드
        :param cnt_to_del: 삭제할 개수
        :param update_stock: 주식 데이터 업데이트 여부
        :return:
        """
        file_path = os.path.join(self.STOCK_DIR, '{}.pkl'.format(corp_code))
        if os.path.isfile(file_path):
            data = pd.read_pickle(file_path)
            if update_stock is True:
                data = data[:-1]
                date_last = data.tail(1)['date'].dt.date.values[0]
                date_next = DateUtils.add_days(date_last)

                new_data = self.crawl_stock(corp_code, date_next)
                if len(new_data.index) > 0:
                    data = data.append(new_data, ignore_index=True)
                    DataUtils.save_pickle(data, file_path)
        else:
            data = self.crawl_stock(corp_code)
            DataUtils.save_pickle(data, file_path)
        if cnt_to_del != 0:
            data = data[:-cnt_to_del]
        return data

    def update_stocks(self):
        corp = CorpLoader()
        corps = corp.get_crops_confidence()
        for row in tqdm(corps.itertuples(index=False), total=len(corps.index), desc="Update Stocks"):
            corp_code = getattr(row, "종목코드")
            self.get_stock_data(corp_code)

    def crawl_stock(self, corp_code: str, start_date: date = None):
        """
        주식 데이터를 크롤링한다.
        :param corp_code:
        :param start_date:
        :return:
        """
        if StockLoader.WEB_SEARCH_FIRST == 'daum':
            search_order_funcs = [self.crawl_stock_daum, self.crawl_stock_naver, self.crawl_stock_daum_before]
        else:
            search_order_funcs = [self.crawl_stock_naver, self.crawl_stock_daum, self.crawl_stock_daum_before]
        for crawl_func in search_order_funcs:
            try:
                return crawl_func(corp_code, start_date)
            except Exception as e:
                if StockLoader.WEB_SEARCH_FIRST == 'daum':
                    StockLoader.WEB_SEARCH_FIRST = 'naver'
                else:
                    StockLoader.WEB_SEARCH_FIRST = 'daum'
                self.logger.error(crawl_func)
                self.logger.error(e)

    def crawl_stock_daum(self, corp_code: str, start_date: date = None) -> pd.DataFrame:
        """
        daum.net 에서 주식 데이터를 크롤링한다.
        :param corp_code: 주식회사 코드
        :param start_date: 시작 날짜
        :return:
        """
        url = "http://finance.daum.net/api/quote/A{code}/days?symbolCode=A{code}&page={page}&perPage=30&pagination=true"
        cols = ['date', 'tradePrice', 'openingPrice', 'highPrice', 'lowPrice', 'accTradeVolume']
        rename_cols = {'date': 'date',
                       'tradePrice': 'close',
                       'openingPrice': 'open',
                       'highPrice': 'high',
                       'lowPrice': 'low',
                       'accTradeVolume': 'volume'}
        headers = {
            'Host': 'finance.daum.net',
            'Referer': 'http://finance.daum.net/quotes/',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        return self.crawl_web_data(corp_code, url, cols, rename_cols, start_date, headers=headers)

    def crawl_stock_naver(self, corp_code: str, start_date: date = None) -> pd.DataFrame:
        """
        Naver 에서 주식데이터를 크롤링한다.
        :param corp_code: 주식회사 코드
        :param start_date: 시작 날짜
        :return:
        """
        url = "http://finance.naver.com/item/sise_day.nhn?code={code}&page={page}"
        cols = ['날짜', '종가', '시가', '고가', '저가', '거래량']
        rename_cols = {'날짜': 'date',
                       '종가': 'close',
                       '시가': 'open',
                       '고가': 'high',
                       '저가': 'low',
                       '거래량': 'volume'}
        return self.crawl_web_data(corp_code, url, cols, rename_cols, start_date, data_type='html', date_col_name='날짜')

    def crawl_stock_daum_before(self, corp_code: str, start_date: date = None) -> pd.DataFrame:
        """
        이전 다음 홈페이지에서 크롤링한다.
        :param corp_code: 주식회사 코드
        :param start_date: 시작 날짜
        :return:
        """
        if date is None:
            url = "http://finance.daum.net/item/quote_yyyymmdd_sub.daum?code={code}&modify=1&page={page}"
        else:
            url = "http://finance.daum.net/item/quote_yyyymmdd.daum?code={code}&page={page}"
        cols = ['일자', '종가', '시가', '고가', '저가', '거래량']
        rename_cols = {'일자': 'date',
                       '종가': 'close',
                       '시가': 'open',
                       '고가': 'high',
                       '저가': 'low',
                       '거래량': 'volume'}
        return self.crawl_web_data(corp_code, url, cols, rename_cols, start_date, data_type='html', date_col_name='일자')

    def crawl_web_data(self, corp_code: str, url: str, cols: list, rename_cols: dict, start_date: date = None,
                       headers: dict = None, data_type: str = 'json', date_col_name: str = 'date'):
        """
        웹 데이터를 크롤링 한다.
        :param corp_code: 주식회사 코드
        :param url: 웹페이지 URL
        :param cols: 가져올 컬럼명들
        :param rename_cols: 바꿀 이름 정보
        :param start_date: 시작 날짜
        :param headers: Request Headers
        :param data_type: 데이터 타입
        :param date_col_name: 날짜 컬럼 이름
        :return:
        """
        data = pd.DataFrame()
        page = 1
        bf_date = None
        while True:
            page_data = self.crawl_web_page(corp_code, url, cols, page, headers, data_type, date_col_name)
            if len(page_data.index) == 0:
                break
            last_date = page_data.tail(1)[date_col_name].dt.date.values[0]
            if not (bf_date is None) and bf_date == last_date:
                break
            data = data.append(page_data, ignore_index=True)
            if not (start_date is None):
                if start_date > last_date:
                    break
            page += 1
            bf_date = last_date

        data = self.remove_unneeded_rows(data, start_date, date_col_name)

        # 정렬 및 컬럼명 변경
        if data.shape[0] != 0:
            data = data.sort_values(by=date_col_name).reset_index(drop=True)
            data.rename(columns=rename_cols, inplace=True)
        return data

    @staticmethod
    def remove_unneeded_rows(data: pd.DataFrame, start_date: date, date_col_name: str = 'date'):
        """
        필요 없는 행 제거
        :param data: 데이터
        :param start_date: 시작 날짜
        :param date_col_name: 날짜 컬럼 이름
        :return:
        """
        if start_date is None:
            return data
        drop_cnt = 0
        df_len = len(data.index)
        for i in range(df_len):
            last_date = data.loc[df_len - i - 1, date_col_name]
            if start_date > last_date:
                drop_cnt += 1
            else:
                break
        if drop_cnt > 0:
            data = data[:-drop_cnt]
        return data

    @staticmethod
    def crawl_web_page(corp_code: str, url: str, cols: list, page: int, headers: dict = None, data_type: str = 'json',
                       date_col_name: str = 'date') -> pd.DataFrame:
        """
        웹 데이터의 한 페이지를 크롤링 한다.
        :param corp_code: 주식회사 코드
        :param url: 웹페이지 URL
        :param cols: 가져올 컬럼명들
        :param page: 페이지 번호
        :param headers: Request Headers
        :param data_type: 데이터 타입
        :param date_col_name: 날짜 컬럼 이름
        :return:
        """
        pg_url = url.format(code=corp_code, page=page)
        if data_type == 'json':
            if not (headers is None):
                response = requests.get(pg_url, headers=headers)
            else:
                response = requests.get(pg_url)
            json = response.json()
            page_data = pd.DataFrame.from_dict(json['data'])
        else:
            page_data = pd.read_html(pg_url, header=0)[0]
            page_data = page_data.dropna()

        if len(page_data.index) != 0:
            page_data = page_data[cols]
            page_data[date_col_name] = pd.to_datetime(page_data[date_col_name].str.slice(0, 10),
                                                      format=DateUtils.DATE_FORMAT)
        return page_data
