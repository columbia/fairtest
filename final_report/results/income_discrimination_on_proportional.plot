set terminal postscript eps enhanced color solid 'Times' 28

set style data histogram
set style histogram rowstack
set style fill pattern

set boxwidth 0.7 absolute
set key invert above width 1 vertical maxrows 1
set key samplen 2.5 spacing 0.85 font ",30"

set xlabel "User-visits grouped by user's income"
set ylabel "Prices shown on each user-visit (%).\n Total: \\~500,000 user-visits"

set border 3
set yrange [0:150]
set xtics nomirror
set ytics nomirror
set ytics ('25' 25, '50' 50, '75' 75, '100' 100)
set xtic rotate by -45 scale 0
set xtics ('< $5,000' 0, '> $5,000' 1, '> $10,000' 2, '> $20,000' 3, '> 40,000' 4,\
           '> $80,000' 5, '> $160,000' 6, '> $320,000   ' 7)
set datafile separator ","
set xlabel offset 0,-2,0

plot newhistogram, "income_discrimination_on_proportional.csv"\
  u 6 t "High price" lc rgbcolor "black" lt 1 fs pattern 3,\
  '' u 5 t "Low price" lc rgbcolor "black" lt 1  fs pattern 1,\
  '' using ($1-1):(105):2 notitle with labels rotate left
