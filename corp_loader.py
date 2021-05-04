import os
from io import BytesIO

import pandas as pd
import requests

import logging_config as log
from data_converter import DataConverter
from utils.data_utils import DataUtils
from utils.date_utils import DateUtils


class CorpLoader:
    """
    주식회사 관련 데이터를 가져온다.
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    # 학습에 포할할 최소 년 간격
    MIN_YEARS_INTERVAL = 5

    def __init__(self):
        self.logger = log.get_logger(self.__class__.__name__)

    def get_eval_crops(self):
        """
        평가를 위한 주식회사 데이터를 100개만 가져온다.
        :return:
        """
        corps = self.get_crops_with_market_cap()
        selected_first = corps[:50]
        selected_last = corps[len(corps.index) - 60:-10]
        return selected_first.append(selected_last, ignore_index=True)

    def get_crops_with_market_cap(self) -> pd.DataFrame:
        """
        시가총액을 포함한 주식회사 데이터를 가져온다.
        :return:
        """
        corps = self.get_corps_all(self.MIN_YEARS_INTERVAL)
        corps_cap = self.get_corps_market_cap()
        corps = corps.merge(corps_cap, on='종목코드')
        corps = corps.sort_values(by=["시가총액"], ascending=False).reset_index(drop=True)
        return corps

    def get_crops_confidence(self) -> pd.DataFrame:
        """
        시가총액을 포함한 주식회사 데이터를 가져온다.
        :return:
        """
        corps = self.get_crops_with_market_cap()
        corps = corps[:-20]
        return corps

    def get_corps_all(self, listing_interval_years: int = 0) -> pd.DataFrame:
        """
        전체 주식회사 데이터를 가져온다.
        :param listing_interval_years: 상장일 현재로부터의 간격
        :return:
        """
        corps = None
        today = DateUtils.get_today_str()
        while True:
            year = today[0:4]
            file_path = os.path.join(self.DATA_DIR, 'corps', year, f'corps_{today}.pkl')
            if os.path.exists(file_path):
                corps = pd.read_pickle(file_path)
            else:
                try:
                    corps = self.crawl_corps_master()
                except Exception as e:
                    self.logger.error(DataUtils.get_error_message(e))
                    self.logger.error(corps)
                    today = DateUtils.to_str_date(DateUtils.add_days(DateUtils.to_date(today), -1))
                    continue
                corps = corps[['종목코드', '회사명', '상장일']]
                converter = DataConverter()
                converter.zfill_stock_code(corps)
                DataUtils.save_pickle(corps, file_path)
            break

        if listing_interval_years != 0:
            today = DateUtils.get_today()
            max_listed_date = DateUtils.add_years(today, -listing_interval_years)
            corps = corps.query(f"상장일<'{max_listed_date}'")
        return corps

    @staticmethod
    def crawl_corps_master() -> pd.DataFrame:
        """
        주식회사 전체 데이터를 크롤링한다.
        :return:
        """
        url = 'https://kind.krx.co.kr/corpgeneral/corpList.do'
        data = {
            'method': 'download',
            'orderMode': '1',  # 정렬컬럼
            'orderStat': 'D',  # 정렬 내림차순
            'searchType': '13',  # 검색유형: 상장법인
            'fiscalYearEnd': 'all',  # 결산월: 전체
            'location': 'all',  # 지역: 전체
        }
        r = requests.post(url, data=data)
        f = BytesIO(r.content)
        dfs = pd.read_html(f, header=0, parse_dates=True)
        return dfs[0]

    def get_corps_market_cap(self):
        """
        시가총액 데이터를 가져온다.
        :return:
        """
        date = DateUtils.get_today_str()
        market_cap = None
        for _ in range(100):
            year = date[0:4]
            file_path = os.path.join(self.DATA_DIR, "corps_market_cap", year, f"market_cap_{date}.pkl")
            if not os.path.isfile(file_path):
                master_data = self.crawl_corps_master_cap(date)
                if master_data is None:
                    d = DateUtils.to_date(date)
                    d = DateUtils.add_days(d, -1)
                    date = DateUtils.to_str_date(d)
                    self.logger.error(f"{date}로 돌림...")
                    continue
                market_cap = master_data[['종목코드', '시가총액']]
                # market_cap.rename(columns={'자본금(원)': '시가총액'}, inplace=True)
                converter = DataConverter()
                converter.zfill_stock_code(market_cap)
                DataUtils.save_pickle(market_cap, file_path)
            else:
                market_cap = pd.read_pickle(file_path)
            break
        return market_cap

    def crawl_corps_master_cap(self, date):
        """
        시가총액 데이터를 크롤링한다.
        :return:
        """
        header = {
            'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201',
            "User-Agent": "Mozilla/5.0"
        }
        gen_otp_url = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
        gen_otp_data = {
            'name': 'fileDown',
            'share': '1',
            'money': '1',
            'csvxls_isNo': 'false',
            'url': 'dbms/MDC/STAT/standard/MDCSTAT01501',
            'trdDd': date.replace("-", ""),
            'mktId': 'ALL'
        }
        r = requests.post(gen_otp_url, gen_otp_data, headers=header)
        down_url = 'http://data.krx.co.kr/comm/fileDn/download_excel/download.cmd'
        down_data = {'code': r.content}
        r = requests.post(down_url, down_data, headers=header)
        df = None
        try:
            df = pd.read_excel(BytesIO(r.content), header=0, thousands=',')
        except Exception as e:
            self.logger.error(DataUtils.get_error_message(e))
            # self.logger.error(r.content)
        return df
