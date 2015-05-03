set terminal postscript eps enhanced color solid 'Times' 28
set xlabel "Dependency of price engine on user's location (%)"
set ylabel "Deltas for sexes"

set key at 65, 0.2

set ytics ('0.1' 0.1, '0.2' 0.2)

set border 3
set yrange [0:0.2]
set xtics nomirror
set ytics nomirror
set datafile separator ","
plot 'sex_discrimination_on_location_dependency.csv'\
  using 1:2 title    "Male" with linespoints lt 1 lw 10,\
  '' using 1:3 title "Female" with linespoints lt 2 lw 10
