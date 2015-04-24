set terminal postscript eps enhanced color solid 'Times' 25
set xlabel "Dependency of price engine on user's location (%)"
set ylabel "#Races discriminated based on user's location. \n Total: 8 races."
set yrange [0:9]

set border 3
set xtics nomirror
set ytics nomirror
set ytics ('0' 0, '2' 2, '4' 4, '6' 6, '8' 8)
set datafile separator ","
#set style histogram errorbars gap 1 lw 3
#set style fill solid 0.5
plot 'race_discrimination_on_location.csv'\
  using 1:3 title "Avg. #races discriminated\n based on user's locaton" with points lc -1 lw 8 pointtype 6,\
  "" using 1:3:2:4:xticlabels(1) notitle with errorbars lc rgb 'gray60' lw 1,\
  "" using 1:3 notitle with line lc -1 lw 3
