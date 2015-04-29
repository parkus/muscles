import reduce
import database
import io
import plot
import utils
import check
import settings

with open(database.target_list) as f:
    stars = f.readlines()
    stars = [s.replace('\r\n', '') for s in stars]
    stars = [s.replace('eps eri', 'v-eps-eri') for s in stars]

