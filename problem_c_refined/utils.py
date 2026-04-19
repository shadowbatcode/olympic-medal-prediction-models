import pycountry
import unicodedata
import re
import numpy as np
from geopy.geocoders import Nominatim
noc_to_full_name = {
    'CHN': 'China', 'DK': 'Denmark', 'NL': 'Netherlands', 'FIN': 'Finland',
    'ROU': 'Romania', 'NOR': 'Norway', 'EST': 'Estonia', 'FRA': 'France',
    'MAR': 'Morocco', 'ESP': 'Spain', 'EGY': 'Egypt', 'IR': 'Iran',
    'BG': 'Bulgaria', 'EG': 'Ecuador', 'SD': 'Sudan', 'RUS': 'Russia',
    'ARG': 'Argentina', 'CUB': 'Cuba', 'BLR': 'Belarus', 'GR': 'Greece',
    'CMR': 'Cameroon', 'TUR': 'Turkey', 'CL': 'Chile', 'MEX': 'Mexico',
    'RU': 'Russia', 'ITA': 'Italy', 'NI': 'Nicaragua', 'HUN': 'Hungary',
    'NGR': 'Nigeria', 'ALG': 'Algeria', 'KUW': 'Kuwait', 'BRN': 'Bahrain',
    'PAK': 'Pakistan', 'IRQ': 'Iraq', 'UAR': 'United Arab Republic',
    'LIB': 'Lebanon', 'QAT': 'Qatar', 'MAS': 'Malaysia', 'AZE': 'Azerbaijan',
    'GER': 'Germany', 'CAN': 'Canada', 'IRL': 'Ireland', 'USA': 'United States',
    'RSA': 'South Africa', 'ERI': 'Eritrea', 'TAN': 'Tanzania', 'JOR': 'Jordan',
    'TUN': 'Tunisia', 'LBA': 'Libya', 'BEL': 'Belgium', 'AUS': 'Australia',
    'DJI': 'Djibouti', 'PLE': 'Palestine', 'COM': 'Comoros', 'KAZ': 'Kazakhstan',
    'BRU': 'Brunei', 'IND': 'India', 'KSA': 'Saudi Arabia', 'SYR': 'Syria',
    'MDV': 'Maldives', 'ETH': 'Ethiopia', 'UAE': 'United Arab Emirates',
    'YAR': 'Yemen Arab Republic', 'INA': 'Indonesia', 'PNL': 'Poland',
    'SGP': 'Singapore', 'UZB': 'Uzbekistan', 'KGZ': 'Kyrgyzstan', 'TJK': 'Tajikistan',
    'JPN': 'Japan', 'CGO': 'Congo', 'RUI': 'Russia', 'BRA': 'Brazil',
    'GDR': 'German Democratic Republic', 'MON': 'Monaco', 'ISR': 'Israel',
    'URU': 'Uruguay', 'SWE': 'Sweden', 'SRI': 'Sri Lanka', 'ARM': 'Armenia',
    'CIV': 'Ivory Coast', 'KEN': 'Kenya', 'BEN': 'Benin', 'GBR': 'Great Britain',
    'GHA': 'Ghana', 'SOM': 'Somalia', 'NIG': 'Niger', 'MLI': 'Mali',
    'AFG': 'Afghanistan', 'POL': 'Poland', 'CRC': 'Costa Rica', 'PAN': 'Panama',
    'GEO': 'Georgia', 'SLO': 'Slovenia', 'GUY': 'Guyana', 'NZL': 'New Zealand',
    'POR': 'Portugal', 'PAR': 'Paraguay', 'ANG': 'Angola', 'VEN': 'Venezuela',
    'COL': 'Colombia', 'FRG': 'Federal Republic of Germany', 'BAN': 'Bangladesh',
    'PER': 'Peru', 'ESA': 'El Salvador', 'UGA': 'Uganda', 'HON': 'Honduras',
    'ECU': 'Ecuador', 'TKM': 'Turkmenistan', 'MRI': 'Mauritius', 'SEY': 'Seychelles',
    'TCH': 'Czechoslovakia', 'LUX': 'Luxembourg', 'MTN': 'Mauritania',
    'SKN': 'Saint Kitts and Nevis', 'TTO': 'Trinidad and Tobago', 'DOM': 'Dominican Republic',
    'VIN': 'Saint Vincent and the Grenadines', 'PUR': 'Puerto Rico', 'JAM': 'Jamaica',
    'LBR': 'Liberia', 'RUR': 'Russia', 'NEP': 'Nepal', 'MGL': 'Mongolia',
    'AUT': 'Austria', 'PLW': 'Palau', 'LTU': 'Lithuania', 'TOG': 'Togo',
    'NAM': 'Namibia', 'AHO': 'Netherlands Antilles', 'UKR': 'Ukraine',
    'ISL': 'Iceland', 'ASA': 'American Samoa', 'SAM': 'Samoa', 'EUN': 'Unified Team',
    'RWA': 'Rwanda', 'CRO': 'Croatia', 'DMA': 'Dominica', 'HAI': 'Haiti',
    'MLT': 'Malta', 'CYP': 'Cyprus', 'GUI': 'Guinea', 'BIZ': 'Belize',
    'YMD': 'Yemen', 'THA': 'Thailand', 'BER': 'Bermuda', 'ANZ': 'Australasia',
    'SCG': 'Serbia and Montenegro', 'SLE': 'Sierra Leone', 'PNG': 'Papua New Guinea',
    'YEM': 'Yemen', 'IOA': 'Independent Olympic Athletes', 'OMA': 'Oman',
    'FIJ': 'Fiji', 'VAN': 'Vanuatu', 'MDA': 'Moldova', 'BAH': 'Bahamas',
    'GUA': 'Guatemala', 'YUG': 'Yugoslavia', 'LAT': 'Latvia', 'SRB': 'Serbia',
    'IVB': 'British Virgin Islands', 'MOZ': 'Mozambique', 'ISV': 'Virgin Islands',
    'CAF': 'Central African Republic', 'MAD': 'Madagascar', 'MAL': 'Malaysia',
    'BIH': 'Bosnia and Herzegovina', 'GUM': 'Guam', 'CAY': 'Cayman Islands',
    'SVK': 'Slovakia', 'BAR': 'Barbados', 'GBS': 'Guinea-Bissau', 'TLS': 'Timor-Leste',
    'COD': 'Democratic Republic of the Congo', 'GAB': 'Gabon', 'SMR': 'San Marino',
    'LAO': 'Laos', 'BOT': 'Botswana', 'ROT': 'Refugee Olympic Team',
    'KOR': 'South Korea', 'CAM': 'Cambodia', 'PRK': 'North Korea',
    'SOL': 'Solomon Islands', 'SEN': 'Senegal', 'CPV': 'Cape Verde',
    'CZE': 'Czech Republic', 'CRT': 'Caribbean Team', 'GEQ': 'Equatorial Guinea',
    'BOL': 'Bolivia', 'SAA': 'Soviet Asian Republics', 'ANT': 'Antigua and Barbuda',
    'AND': 'Andorra', 'ZIM': 'Zimbabwe', 'GRN': 'Grenada', 'HKG': 'Hong Kong',
    'LCA': 'Saint Lucia', 'FSM': 'Micronesia', 'MYA': 'Myanmar', 'MAW': 'Malawi',
    'ZAM': 'Zambia', 'RHO': 'Rhodesia', 'TPE': 'Chinese Taipei', 'STP': 'Sao Tome and Principe',
    'MKD': 'North Macedonia', 'BOH': 'Bohemia', 'LIE': 'Liechtenstein',
    'MNE': 'Montenegro', 'GAM': 'Gambia', 'COK': 'Cook Islands', 'ALB': 'Albania',
    'WIF': 'West Indies Federation', 'SWZ': 'Eswatini', 'BUR': 'Burkina Faso',
    'NBO': 'Nairobi', 'BDI': 'Burundi', 'ARU': 'Aruba', 'NRU': 'Nauru',
    'VNM': 'Vietnam', 'VIE': 'Vietnam', 'BHU': 'Bhutan', 'MHL': 'Marshall Islands',
    'KIR': 'Kiribati', 'UNK': 'Unknown', 'TUV': 'Tuvalu', 'TGA': 'Tonga',
    'NFL': 'Newfoundland', 'KOS': 'Kosovo', 'SSD': 'South Sudan', 'LES': 'Lesotho',
    'ROC': 'Russian Olympic Committee', 'EOR': 'Refugee Olympic Team',
    'LBN': 'Lebanon', 'AIN': 'Independent Olympic Athletes'
}
country_coordinates = {
    "China": (35.8617, 104.1954),
    "Denmark": (56.2639, 9.5018),
    "Netherlands": (52.1326, 5.2913),
    "Finland": (61.9241, 25.7482),
    "Romania": (45.9432, 24.9668),
    "Norway": (60.4720, 8.4689),
    "Estonia": (58.5953, 25.0136),
    "France": (46.6034, 1.8883),
    "Morocco": (31.7917, -7.0926),
    "Spain": (40.4637, -3.7492),
    "Egypt": (26.8206, 30.8025),
    "Iran": (32.4279, 53.6880),
    "Bulgaria": (42.7339, 25.4858),
    "Ecuador": (-1.8312, -78.1834),
    "Sudan": (12.8628, 30.2176),
    "Russia": (61.5240, 105.3188),
    "Argentina": (-38.4161, -63.6167),
    "Cuba": (21.5218, -77.7812),
    "Belarus": (53.7098, 27.9534),
    "Greece": (39.0742, 21.8243),
    "Cameroon": (7.3697, 12.3547),
    "Turkey": (38.9637, 35.2433),
    "Chile": (-35.6751, -71.5430),
    "Mexico": (23.6345, -102.5528),
    "Italy": (41.8719, 12.5674),
    "Nicaragua": (12.8654, -85.2072),
    "Hungary": (47.1625, 19.5033),
    "Nigeria": (9.0820, 8.6753),
    "Algeria": (28.0339, 1.6596),
    "Kuwait": (29.3117, 47.4818),
    "Bahrain": (26.0667, 50.5577),
    "Pakistan": (30.3753, 69.3451),
    "Iraq": (33.2232, 43.6793),
    "United States": (37.0902, -95.7129),
    "Germany": (51.1657, 10.4515),
    "South Africa": (-30.5595, 22.9375),
    "India": (20.5937, 78.9629),
    "Japan": (36.2048, 138.2529),
    "Brazil": (-14.2350, -51.9253),
    "Australia": (-25.2744, 133.7751),
    "Canada": (56.1304, -106.3468),
    "United Arab Emirates": (23.4241, 53.8478),
    "Singapore": (1.3521, 103.8198),
    "Indonesia": (-0.7893, 113.9213),
    "United Kingdom": (55.3781, -3.4360),
    "South Korea": (35.9078, 127.7669),
    "New Zealand": (-40.9006, 174.8860),
    "Malaysia": (4.2105, 101.9758),
    "Thailand": (15.8700, 100.9925),
    "Vietnam": (14.0583, 108.2772),
    "Israel": (31.0461, 34.8516),
    "Saudi Arabia": (23.8859, 45.0792),
    "Qatar": (25.3548, 51.1839),
    "Kyrgyzstan": (41.2044, 74.7661),
    "Tajikistan": (38.8610, 71.2761),
    "Uzbekistan": (41.3775, 64.5853),
    "Kazakhstan": (48.0196, 66.9237),
    "Turkey": (38.9637, 35.2433),
    "France": (46.6034, 1.8883),
    "Ukraine": (48.3794, 31.1656),
}

def clean_text(text):
    # 清理乱码字符
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
    return cleaned_text
def normalize_text(text):
    # 规范化特殊字符
    normalized_text = unicodedata.normalize('NFKD', text)
    return normalized_text

def noc_to_country(noc):
    global noc_to_full_name
    return noc_to_full_name[noc]

def process_host_data(host):
    if 'Cancelled' in host:
        return None
    host_list = host.split(',')
    cleaned_host_list = [re.sub(r'\(.*?\)', '', h).strip() for h in host_list]
    return cleaned_host_list[1]

def normalize(df):
    df_normalized = (df - df.min()) / (df.max() - df.min()) * 100
    return df_normalized

def map_coordinates(full_names):
    global country_coordinates
    if full_names in country_coordinates:
        site = country_coordinates[full_names]
        return site
    else:
        return None, None

