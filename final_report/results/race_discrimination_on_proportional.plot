set terminal postscript eps enhanced color solid 'Times' 30

set style data histogram
set style histogram rowstack
set style fill pattern

set boxwidth 0.7 absolute
set key outside
set key samplen 1.5 spacing 1.85

set xlabel "User-visits grouped by user's race"
set ylabel "Prices shown on each user-visit (%).\n Total: 1,000,000 user-visits"

set border 3
set yrange [0:135]
set xtics nomirror
set ytics nomirror
set ytics ('25' 25, '50' 50, '75' 75, '100' 100)
set xtic rotate by -45 scale 0

set xtics ('White' 0, 'Hispanic' 1, 'Afroamerican' 2, 'Indian, Alaskan' 3, 'Asian' 4,\
           'Pasific Islander' 5, 'Other' 6, 'Two or More' 7)
set datafile separator ","
set xlabel offset 0,2,0
set ylabel offset 0,-3,0

plot newhistogram, "race_discrimination_on_proportional.csv"\
  u 6 t "High" lc rgbcolor "black" lt 1 fs pattern 3,\
  '' u 5 t "Low" lc rgbcolor "black" lt 1  fs pattern 1,\
  '' using ($1-1):(105):2 notitle with labels rotate left
