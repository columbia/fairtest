set terminal postscript eps enhanced color solid 'Times' 28
set xlabel "Dependency of price engine on user's location (%)"
set ylabel "Deltas for race."

set border 3
set yrange [0:1]
set xtics nomirror
set ytics nomirror
set datafile separator ","
plot 'delta_race.csv'\
  using 1:2 title "race-1" with lines lt 1,\
  '' using 1:3 title "race-2" with lines lt 2,\
  '' using 1:4 title "race-3" with lines lt 3,\
  '' using 1:5 title "race-4" with lines lt 4,\
  '' using 1:6 title "race-5" with lines lt 5,\
  '' using 1:7 title "race-6" with lines lt 6,\
  '' using 1:8 title "race-7" with lines lt 7,\
  '' using 1:9 title "race-8" with lines lt 8
