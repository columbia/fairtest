set terminal postscript eps enhanced color solid 'Times' 28
set xlabel "Dependency of price engine on user's location (%)"
set ylabel "Deltas for incomes"

set border 3
set yrange [0:1]
set xtics nomirror
set ytics nomirror
set datafile separator ","
plot 'temp_income.csv'\
  using 1:2 title "income-1" with lines lt 1,\
  '' using 1:3 title "income-2" with lines lt 2,\
  '' using 1:4 title "income-3" with lines lt 3,\
  '' using 1:5 title "income-4" with lines lt 4,\
  '' using 1:6 title "income-5" with lines lt 5,\
  '' using 1:7 title "income-6" with lines lt 6,\
  '' using 1:8 title "income-7" with lines lt 7,\
  '' using 1:9 title "income-8" with lines lt 8
