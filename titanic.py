import pandas as pd

# Generar un DataFrame con los datos del fichero.
tablas_titanic = pd.read_csv('titanic.csv', index_col=0)

print(tablas_titanic)

# Mostrar por pantalla las dimensiones del DataFrame, el número de datos que contiene, los nombres de sus columnas y filas, los tipos de datos de las columnas, las 10 primeras filas y las 10 últimas filas.
print('Dimensiones:', tablas_titanic.shape)
print('Nombres de columnas:', tablas_titanic.columns)
print('Tipos de datos:\n', tablas_titanic.dtypes)
print('Primeras 10 filas:\n', tablas_titanic.head(10))
print('Nombres de filas:', tablas_titanic.index)
print('Últimas 10 filas:\n', tablas_titanic.tail(10))
print('Número de elemntos:', tablas_titanic.size)

#Mostrar por pantalla el porcentaje de personas que si vivieron para contarla
print(tablas_titanic.groupby('Pclass')['Survived'].value_counts(normalize=True))

# Mostrar los nombres de las personas fifi
print(tablas_titanic[tablas_titanic["Pclass"]==1]['Name'].sort_values())

# Mostrar por pantalla el porcentaje de personas que sobrevivieron y que estiraron la pata
print(tablas_titanic['Survived'].value_counts()/tablas_titanic['Survived'].count() * 100)

# Alternativa
print(tablas_titanic['Survived'].value_counts(normalize=True) * 100)

# Eliminar del DataFrame los pasajeros con edad anonima.
tablas_titanic.dropna(subset=['Age'])

# Mostrar la edad media de las mujeres que viajaban en cada clase.
print(tablas_titanic.groupby(['Pclass','Sex'])['Age'].mean().unstack()['female'])

# Añadir una nueva columna booleana para ver si el pasajero era menor de edad .
tablas_titanic['Young'] = tablas_titanic['Age'] < 18

# Mostrar por pantalla los datos del pasajero con id 148
print(tablas_titanic.loc[148])

# Mostrar por pantalla las filas pares del DataFrame.
print(tablas_titanic.iloc[range(0,tablas_titanic.shape[0],2)])