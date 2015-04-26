set terminal postscript eps enhanced color solid 'Times' 25
set xlabel "Dependency of price engine on user's location (%)"
set ylabel "#Sexes discriminated based on user's location. \n Total: 8 sexs."
set yrange [0:3]

set key samplen 2.5 spacing 0.85 font ",30"
set border 3
set xtics nomirror
set ytics nomirror
set ytics ('1' 1, '2' 2)
set datafile separator ","
plot 'sex_discrimination_on_location_dependency.csv'\
  using 1:3:2:4:xticlabels(1) notitle with errorbars lc rgb 'gray60' lw 1,\
  "" using 1:3 title "Avg. #sexs discriminated\n based on user's location" with linespoints lc -1 lw 8 pointtype 16
