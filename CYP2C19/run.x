#if [ -f "dataset.smi" ]; then
#    rm dataset.smi
#fi

#awk -F ',' '{print $4,"\t",$3}' AID_1671197_datatable_all.csv >>  dataset.smi
#tail +7 dataset.smi > oo
#mv oo dataset.smi
#head -n -1 dataset.smi > oo 
#mv oo dataset.smi
#cat dataset.smi |  sort -u  >> oo 
#rm dataset.smi
#tail +2 oo >> dataset.smi
#rm oo

#if [ -f "target.csv" ]; then
#    rm target.csv
#fi

#awk -F ',' '{print $3,"\t",$5}' AID_1671197_datatable_all.csv >> target.csv
#tail +7 target.csv > oo
#mv oo target.csv
#head -n -1 target.csv  > oo
#rm target.csv
#cat oo | sort -u >> target.csv
#rm oo
#grep -v Inconclusive target.csv  >> oo
#mv oo target.csv
#sed -i -e "s/Active/1/" target.csv
#sed -i -e "s/Inactive/0/" target.csv
#sed -i -e "s/\t/,/" target.csv
#sed -i -e "s/ //g" target.csv
#head -n -2 target.csv >> oo
#rm target.csv
#echo -e "Molecule,Target" >> target.csv
#cat oo >> target.csv
#rm oo

python3 make_model.py dataset.rdkit_dscriptors.csv target.csv
rm emissions.csv
python3 make_model.py dataset.morgan_ecfp.csv target.csv
rm emissions.csv

