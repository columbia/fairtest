set terminal postscript eps enhanced color solid 'Times' 28
set xlabel "Dependency of price engine on user's location (%)"
set ylabel "#Incomes discriminated on user's location. \n Total: 8 income groups."
set yrange [0:9]

set key samplen 2.5 spacing 0.85 font ",30"
set border 3
set xtics nomirror
set ytics nomirror
set ytics ('2' 2, '4' 4, '6' 6, '8' 8)
set datafile separator ","
plot 'income_discrimination_on_location_dependency.csv'\
  using 1:3:2:4:xticlabels(1) notitle with errorbars lc rgb 'gray60' lw 1,\
  "" using 1:3 title "Avg. #incomes discriminated\n on user's location" with linespoints lc -1 lw 8 pointtype 16
