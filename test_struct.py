import numpy as np
from numba import njit

# Definieren Sie den benutzerdefinierten Datentyp
struct_dtype = np.dtype([('field1', np.int32), ('field2', np.float32), ('field3', np.int32)])

# Funktion zum Erstellen und Füllen des Arrays
@njit
def create_and_fill_array(size):
    array = np.zeros(size, dtype=struct_dtype)
    return array

# Funktion zum Setzen eines Elements
@njit
def set_element(array, index, field1_value, field2_value, field3_value):
    # Diese Zeilen zusammenfassen in einer Art, die nopython mode kompatibel ist.
    array[index] = (field1_value, field2_value, field3_value)
    #array[index]['field1'] = field1_value
    #array[index]['field2'] = field2_value
    #array[index]['field3'] = field3_value

# Erstellen und Füllen des Arrays
size = 10  # Größe des Arrays
array = create_and_fill_array(size)

# Setzen eines Elements im Array
set_element(array, 3, 42, 3.14, 7)  # Setzt das Element an Index 3 auf (42, 3.14, 7)

# Ausgabe des Arrays zur Überprüfung
for item in array:
    print(item['field1'], item['field2'], item['field3'])

