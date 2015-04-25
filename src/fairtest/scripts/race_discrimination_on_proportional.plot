set terminal postscript eps enhanced color solid 'Times' 25

set style data histogram
set style histogram rowstack
set style fill pattern

set boxwidth 0.7 absolute
set key invert above width 1 vertical maxrows 1
set key samplen 2.5 spacing 0.85 font ",30"

set xlabel "User-visits grouped by race (8 different race groups)"
set ylabel "Distribution of prices shown to user visits(%).\n Total: \\~500,000 user visits"

set border 3
set format "%'g
set yrange [0:120]
set xtics nomirror
set ytics nomirror
set ytics ('25' 25, '50' 50, '75' 75, '100' 100)
set datafile separator ","

plot newhistogram, "race_discrimination_on_proportional.csv"\
  u 5:xticlabels(1) t "Low price" lc rgbcolor "black" lt 1 fs pattern 3,\
  '' u 6:xticlabels(1) t "High price" lc rgbcolor "black" lt 1  fs pattern 1,\
  '' using ($1-1):(102):2 notitle with labels rotate left
