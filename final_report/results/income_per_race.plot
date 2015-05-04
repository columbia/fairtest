set terminal postscript eps enhanced color solid 'Times' 27

set style data histogram
set style histogram rowstack
set style fill pattern

set boxwidth 0.7 absolute
set key invert above width 1 vertical maxrows 1
set key samplen 2.5 spacing 0.85 font ",30"

set xlabel "Users grouped by race"
set ylabel "Average income per race (dollars)"

set border 3
set format "%'g
set yrange [0:110000]
set xtics nomirror
set ytics nomirror
set xtic rotate by -45 scale 0
set datafile separator ","
set xlabel offset 0,2,0
set ytics ('$20,000' 20000, '$40,000' 40000, '$60,000' 60000, '$80,000' 80000, '$100,000' 100000)
set xtics ('White' 0, 'Hispanic' 1, 'Afroamerican' 2, 'Indian or Alaskan' 3,\
           'Asian' 4, 'Pasific Islander' 5, 'Other' 6, ' Two or More' 7)
plot newhistogram, "income_per_race.csv"\
  u 2 t "Average income per race" lc rgbcolor "black" lt 1 fs pattern 3,\
  '' using ($1 - 1):($2 + 2000):3 notitle with labels rotate left
