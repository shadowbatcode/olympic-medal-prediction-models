import wbgapi as wb
import pandas as pd

# Indicador de PIB y país
indicator = 'NY.GDP.MKTP.CD'  # Producto Interno Bruto (PIB) en dólares corrientes
country = 'all'               # Todos los países
time = list(range(1896, 1968))  # Rango de fechas desde 1960 hasta 2023

# Descargar datos del PIB para el rango de fechas
gdp_data = pd.DataFrame()

# Hacer la descarga año por año para evitar problemas de descarga con la API
for year in time:
    yearly_data = wb.data.DataFrame(indicator, country, year, labels=True)
    yearly_data['year'] = year
    gdp_data = pd.concat([gdp_data, yearly_data])

# Asegurar que la columna del PIB es numérica
gdp_data['NY.GDP.MKTP.CD'] = pd.to_numeric(gdp_data['NY.GDP.MKTP.CD'], errors='coerce')

# Formatear el PIB con separadores de miles y punto como separador decimal
gdp_data['NY.GDP.MKTP.CD'] = gdp_data['NY.GDP.MKTP.CD'].apply(lambda x: '{:,.2f}'.format(x).replace(',', '.') if pd.notnull(x) else x)

# Mostrar los datos
print(gdp_data)