#!/bin/sh
# Per-strip three-phase matrices for blind HEIGHT + TILT control of the FIREFLY office.
# Run from Implementation_Radiance/office (Radiance >= 6.1 with rxfluxmtx, spectral -cs 20).
# Needs: weatherdata/AUT_Innsbruck.111200_IWEC.epw, atmos_data/ (from office_with_results),
#        spectra/BauIng_518_smult_780-380.txt (made by _office_step01.sh).
# Settings used for the delivered data (2 cores); on LEO5 raise -c / -ad for lower noise.
np=${NP:-2}
N=${N:-5}
set -e
mkdir -p strips mtx/strips window/BSDF

echo "1) window split into $N horizontal strips (strip 1 = top)"
python3 make_strips.py $N

echo "2) slat-tilt BSDFs (genBSDF blind layer calibrated to the measured +75 file, layered with XXX glazing)"
( cd window/BSDF && cp ../../bsdf_tools.py ../../gen_tilts.py . && python3 gen_tilts.py )
C=`rcalc -e '$1=0;$2=$1;$3=0' spectra/BauIng_518_smult_780-380.txt`
for t in 00 15 30 45 60 75; do rmtxop -ff -c $C window/BSDF/Bauing_model_tilt$t.xml > mtx/tilt$t.mtx; done
rmtxop -ff -c $C window/BSDF/Bauing_ILM_Lamelle_XXX.xml > mtx/noblinds.mtx
rmtxop -ff -c $C window/BSDF/Bauing_ILM_Lamelle_+75.xml > mtx/blinds+75.mtx

echo "3) sky matrix (identical to _office_step02.sh)"
sed -e '/^[12][9012][0-9][0-9],[1-9][0-9]*,[1-9][0-9]*,[0-7],/d' \
    -e '/^[12][9012][0-9][0-9],[1-9][0-9]*,[1-9][0-9]*,19,/d' \
    -e '/^[12][9012][0-9][0-9],[1-9][0-9]*,[1-9][0-9]*,2[0-4],/d' \
    weatherdata/AUT_Innsbruck.111200_IWEC.epw > weatherdata/AUT_Innsbruck.111200_IWECday.epw
gensdaymtx -n $np -v -m 4 -of weatherdata/AUT_Innsbruck.111200_IWECday.epw > mtx/Innsbruck_daytime_sky4.mtx

echo "4) view matrices per strip: workplane, ceiling sensor, seat eye illuminance"
rxfluxmtx -faf -n $np -I+ -cs 20 -ab 5 -ad 25600 -lw 4e-7 -o mtx/strips/wp_%s.mtx \
    -y 441 - strips/window_strips$N.rad base_model.rad < pts/workplane.pts
rxfluxmtx -fo -faf -I+ -dj .8 -ss 16 -st .01 -ab 6 -ad 102400 -lw 1e-8 -lr -10 -cs 20 -n $np \
    -o mtx/strips/ceil_%s.vec -y 1 - strips/window_strips$N.rad base_model.rad < pts/ceil_sensor.pts
rxfluxmtx -faf -n $np -I+ -cs 20 -ab 5 -ad 25600 -lw 4e-7 -o mtx/strips/seat_%s.mtx \
    -y 3 - strips/window_strips$N.rad base_model.rad < pts/seats_eye.pts

echo "5) daylight matrix per strip"
k=1
while [ $k -le $N ]; do
  rxfluxmtx -n $np -c 2000 -o mtx/strips/daylight_strip$k.mtx -ab 3 -ad 8192 -lw 1e-6 -cs 20 \
      -ff strips/strip$k.rad sources/sky_r4.rad base_model.rad
  k=`expr $k + 1`
done

echo "6) env data"
python3 build_v2_data.py ../office_with_results env_v2_data
