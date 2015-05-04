set terminal postscript eps enhanced color solid 'Times' 30 

set style data histogram
set style histogram rowstack
set style fill pattern

set boxwidth 0.7 absolute
set key outside
set key samplen 1.5 spacing 1.85


set xlabel "Online visits grouped by user's sex"
set ylabel "Prices shown upon each visit (%)\n Total: 1,000,000 user-visits"

set border 3
set format "%'g
set yrange [0:135]
set xtics nomirror
set ytics nomirror
set ytics ('25' 25, '50' 50, '75' 75, '100' 100)
set xtic rotate by -45 scale 0
set xtics ('Male   ' 0, 'Female    ' 1)
set datafile separator ","
set xlabel offset 0,1,0
set ylabel offset 0,-3,0

plot newhistogram, "sex_discrimination_on_proportional.csv"\
  u 6:xticlabels(7) t "High" lc rgbcolor "black" lt 1 fs pattern 3,\
  '' u 5:xticlabels(7) t "Low" lc rgbcolor "black" lt 1  fs pattern 1,\
  '' using ($1):(110):2 notitle with labels
#rotate left
