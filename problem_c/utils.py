import pycountry
import unicodedata
import re
import numpy as np
def clean_text(text):
    # 清理乱码字符
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
    return cleaned_text
def normalize_text(text):
    # 规范化特殊字符
    normalized_text = unicodedata.normalize('NFKD', text)
    return normalized_text

def noc_to_country(noc):
    try:
        # 使用 alpha_3 或 alpha_2 查找国家
        country = pycountry.countries.get(alpha_3=noc) or pycountry.countries.get(alpha_2=noc)
        return country.name if country else noc
    except Exception as e:
        return noc
def process_host_data(host):
    if 'Cancelled' in host:
        return None
    host_list = host.split(',')
    cleaned_host_list = [re.sub(r'\(.*?\)', '', h).strip() for h in host_list]
    return cleaned_host_list[1]


def normalize(df):
    df_normalized = (df - df.min()) / (df.max() - df.min()) * 100
    return df_normalized


