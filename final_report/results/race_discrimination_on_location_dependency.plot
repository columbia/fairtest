set terminal postscript eps enhanced color solid 'Times' 30 
set xlabel "Dependency of price engine on user's location (%)"
set ylabel "Delta for each race"
set key at 65, 0.3
set border 3
set yrange [0:0.3]
set xtics nomirror
set ytics nomirror
set ytics ('0' 0, '0.05' 0.05, '0.2' 0.2)
set arrow from 0,0.05 to 100,0.05 nohead lt 0 lw 2
set datafile separator ","

plot 'race_discrimination_on_location_dependency.csv'\
  using 1:2 title "White" with linespoints lt 1,\
  '' using 1:3 title "Hispanic" with linespoints lt 2,\
  '' using 1:4 title "Afroamerican" with linespoints lt 3,\
  '' using 1:5 title "Indian or Alaskan" with linespoints lt 4,\
  '' using 1:6 title "Asian" with linespoints lt 5,\
  '' using 1:7 title "Pacific Islander" with linespoints lt 6,\
  '' using 1:8 title "Other" with linespoints lt 7,\
  '' using 1:9 title "Two or More" with linespoints lt 8
