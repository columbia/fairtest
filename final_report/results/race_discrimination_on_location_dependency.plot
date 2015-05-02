set terminal postscript eps enhanced color solid 'Times' 28
set xlabel "Dependency of price engine on user's location (%)"
set ylabel "Delta for each race"

set border 3
set yrange [0:0.2]
set xtics nomirror
set ytics nomirror
set datafile separator ","
plot 'race_discrimination_on_location_dependency.csv'\
  using 1:2 title "White" with lines lt 1,\
  '' using 1:3 title "Hispanic" with lines lt 2,\
  '' using 1:4 title "Afroamerican" with lines lt 3,\
  '' using 1:5 title "Indian or Alaskan" with lines lt 4,\
  '' using 1:6 title "Asian" with lines lt 5,\
  '' using 1:7 title "Pacific Islander" with lines lt 6,\
  '' using 1:8 title "Other" with lines lt 7,\
  '' using 1:9 title "Two or More" with lines lt 8
