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
    MIN_YEARS_INTERVAL = 10

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
                    self.logger.error(e)
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
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do'
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
        dfs = pd.read_html(f, header=0, parse_dates=['상장일'])
        return dfs[0]

    def get_corps_market_cap(self):
        """
        시가총액 데이터를 가져온다.
        :return:
        """
        date = DateUtils.get_today_str()
        year = date[0:4]
        file_path = os.path.join(self.DATA_DIR, "corps_market_cap", year, "market_cap_{}.pkl".format(date))
        if not os.path.isfile(file_path):
            master_data = self.crawl_corps_master_cap()
            market_cap = master_data[['종목코드', '자본금(원)']]
            market_cap.rename(columns={'자본금(원)': '시가총액'}, inplace=True)
            converter = DataConverter()
            converter.zfill_stock_code(market_cap)
            DataUtils.save_pickle(market_cap, file_path)
        else:
            market_cap = pd.read_pickle(file_path)
        return market_cap

    @staticmethod
    def crawl_corps_master_cap():
        """
        시가총액 데이터를 크롤링한다.
        :return:
        """
        header = {
            'Referer': 'https://marketdata.krx.co.kr/mdi',
            "User-Agent": "Mozilla/5.0"
        }
        gen_otp_url = 'https://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
        gen_otp_data = {
            'name': 'fileDown',
            'filetype': 'xls',
            'url': 'MKD/04/0406/04060100/mkd04060100_01',
            'market_gubun': 'ALL',  # 시장구분: ALL=전체
            'pagePath': '/contents/MKD/04/0406/04060100/MKD04060100.jsp',
            'sort_type': 'A',
            'lst_stk_vl': 1,
            'cpt': 1,
            'isu_cdnm': '전체'
        }
        r = requests.post(gen_otp_url, gen_otp_data, headers=header)
        down_url = 'http://file.krx.co.kr/download.jspx'
        down_data = {'code': r.content}
        r = requests.post(down_url, down_data, headers=header)
        df = pd.read_excel(BytesIO(r.content), header=0, thousands=',')
        return df
