set terminal postscript eps enhanced color solid 'Times' 25

set style data histogram
set style histogram rowstack
set style fill pattern

set boxwidth 0.7 absolute
set key invert above width 1 vertical maxrows 1
set key samplen 2.5 spacing 0.85 font ",30"

set xlabel "Users grouped by race"
set ylabel "Average income per race"

set border 3
set format "%'g
set yrange [0:*]
set xtics nomirror
set ytics nomirror
set xtic rotate by -45 scale 0
set datafile separator ","
set xlabel offset 0,-2,0

plot newhistogram, "income_per_race.csv"\
  u 2:xticlabels(4) t "Average income per race" lc rgbcolor "black" lt 1 fs pattern 3,\
  '' using ($1 - 1):($2 + 1000):3 notitle with labels rotate left
