set terminal postscript eps enhanced color solid 'Times' 30

set style data histogram
set style histogram rowstack
set style fill pattern

set boxwidth 0.7 absolute
set key outside
#invert above width 1 vertical maxrows 1
set key samplen 1.5 spacing 1.85

set xlabel "User-visits grouped by user's income"
set ylabel "Prices shown on each user-visit (%).\n Total: 1,000,000 user-visits"

set border 3
set yrange [0:130]
set xtics nomirror
set ytics nomirror
set ytics ('25' 25, '50' 50, '75' 75, '100' 100)
set xtic rotate by -45 scale 0

set xtics ('< $5,000' 0, '> $5,000' 1, '> $10,000' 2, '> $20,000' 3, '> 40,000' 4,\
           '> $80,000' 5, '> $160,000' 6, '> $320,000   ' 7)
set datafile separator ","
set xlabel offset 0,1.5,0
set ylabel offset 0,-3,0

plot newhistogram, "income_discrimination_on_proportional.csv"\
  u 6 t "High" lc rgbcolor "black" lt 1 fs pattern 3,\
  '' u 5 t "Low" lc rgbcolor "black" lt 1  fs pattern 1,\
  '' using ($1-1):(105):2 notitle with labels rotate left
