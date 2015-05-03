set terminal postscript eps enhanced color solid 'Times' 28
set xlabel "Dependency of price engine on user's location (%)"
set ylabel "Deltas for incomes"

set key at 70,0.3

set border 3
set yrange [0:0.3]
set xtics nomirror
set ytics nomirror
set ytics ('0' 0, '0.1' 0.1, '0.2' 0.2)
set datafile separator ","
plot 'income_discrimination_on_location_dependency.csv'\
  using 1:2 title    "Income < $5,000" with linespoints lt 1,\
  '' using 1:3 title "Income > $5,000" with linespoints lt 2,\
  '' using 1:4 title "Income > $10,000" with linespoints lt 3,\
  '' using 1:5 title "Income > $20,000" with linespoints lt 4,\
  '' using 1:6 title "Income > $40,000" with linespoints lt 5,\
  '' using 1:7 title "Income > $80,000" with linespoints lt 6,\
  '' using 1:8 title "Income > $160,000" with linespoints lt 7,\
  '' using 1:9 title "Income > $320,000" with linespoints lt 8
