#!/bin/bash

INPUT=nrj_st1

mkdir hNOX5_mbH_${INPUT}
cd hNOX5_mbH_${INPUT}

# reinitialize module
module purge

# modules loading
module load namd/2.13-mpi-charmpp-mpi
#module load namd/2.14

cp ../start/eq_st1/step5_input.str .
cp ../start/eq_st1/step5_input.colvar.str .
cp ../psf/*.psf .
cp ../start/eq_st1/step5_assembly.pdb .
cp ../start/eq_st1/step5_assembly.crd .
cp -r ../start/eq_st1/toppar .

# START and END  refer to the  first and last MD runs to be analyzed
START=1
END=40

for ((RUN=$START;RUN<=$END;RUN+=1)) ; do  # loop on MD runs

        # Copy the vel coor and xsc files or MD run $RUN
	cp simulation_folder/nox5_MD_st1_${RUN}.vel .
	cp simulation_folder/nox5_MD_st1_${RUN}.coor .
	cp simulation_folder/nox5_MD_st1_${RUN}.xsc .
  
        # Run single point energy calculations along the dynamics
	
	# Décomposition intern sphere (heme 1 and 2)
        cp ../${INPUT}.inp .
        cat  ${INPUT}.inp | sed "s/PSFPSFPSF/nox5st1_heme1_2.psf/" > tmp1
       	cat tmp1 | sed "s/_NUM/_${RUN}/" > tmp2
        cat tmp2 | sed "s/XXXXXX/nrj_st1_e1_heme1_2_${RUN}/"   > tmp3
	mv tmp3 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_heme1_2_e1.out${RUN}
 
        cat  ${INPUT}.inp | sed "s/nox5st1_heme1_2.psf/nox5st1_heme1.psf/" > tmp3
       	cat tmp3 | sed "s/nrj_st1_e1_heme1_2_${RUN}/nrj_st1_e1_heme1_${RUN}/"  > tmp4
	mv tmp4 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_heme1_e1.out${RUN}

        cat  ${INPUT}.inp | sed "s/nox5st1_heme1.psf/nox5st1_heme2.psf/" > tmp5
	cat tmp5 | sed "s/nrj_st1_e1_heme1_${RUN}/nrj_st1_e1_heme2_${RUN}/"  > tmp6
	mv tmp6 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_heme2_e1.out${RUN} 
	
	# Fad contribution 
        cat  ${INPUT}.inp | sed "s/nox5st1_heme2.psf/nox5st1_fad.psf/" > tmp7
       	cat tmp7 | sed "s/nrj_st1_e1_heme2_${RUN}/nrj_st1_e1_fad_${RUN}/"  > tmp8
	mv tmp8 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_fad_e1.out${RUN}
	
	# Upper leaflet POPE contribution
        cat  ${INPUT}.inp | sed "s/nox5st1_fad.psf/nox5st1_lipid_pope_upper_leaflet.psf/" > tmp9
       	cat tmp9 | sed "s/nrj_st1_e1_fad_${RUN}/nrj_st1_e1_POPE_upper_${RUN}/"  > tmp10
	mv tmp10 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_POPE_upper_e1.out${RUN}
        
	# Upper leaflet POPG contribution
	cat  ${INPUT}.inp | sed "s/nox5st1_lipid_pope_upper_leaflet.psf/nox5st1_lipid_popg_upper_leaflet.psf/" > tmp11
       	cat tmp11 | sed "s/nrj_st1_e1_POPE_upper_${RUN}/nrj_st1_e1_POPG_upper_${RUN}/"  > tmp12
	mv tmp12 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_POPG_upper_e1.out${RUN}
        
	# Lower leaflet POPE contribution
	cat  ${INPUT}.inp | sed "s/nox5st1_lipid_popg_upper_leaflet.psf/nox5st1_lipid_pope_lower_leaflet.psf/" > tmp13
       	cat tmp13 | sed "s/nrj_st1_e1_POPG_upper_${RUN}/nrj_st1_e1_POPE_lower_${RUN}/"  > tmp14
	mv tmp14 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_POPE_lower_e1.out${RUN}
	
	# Lower leaflet POPG contribution
	cat  ${INPUT}.inp | sed "s/nox5st1_lipid_pope_lower_leaflet.psf/nox5st1_lipid_popg_lower_leaflet.psf/" > tmp15
       	cat tmp15 | sed "s/nrj_st1_e1_POPE_lower_${RUN}/nrj_st1_e1_POPG_lower_${RUN}/"  > tmp16
	mv tmp16 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_POPG_lower_e1.out${RUN}
	
	# TM protein contribution
	cat  ${INPUT}.inp | sed "s/nox5st1_lipid_popg_lower_leaflet.psf/nox5st1_protein_tm_part.psf/" > tmp17
       	cat tmp17 | sed "s/nrj_st1_e1_POPE_lower_${RUN}/nrj_st1_e1_TM_protein_${RUN}/"  > tmp18
	mv tmp18 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_TM_protein_e1.out${RUN}
	
	# DH protein contribution
	cat  ${INPUT}.inp | sed "s/nox5st1_protein_tm_part.psf/nox5st1_protein_dh_part.psf/" > tmp19
       	cat tmp19 | sed "s/nrj_st1_e1_TM_protein_${RUN}/nrj_st1_e1_DH_protein_${RUN}/"  > tmp20
	mv tmp20 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_DH_protein_e1.out${RUN}
	
	# NA ions contribution
	cat  ${INPUT}.inp | sed "s/nox5st1_protein_dh_part.psf/nox5st1_counterions_na.psf/" > tmp21
       	cat tmp21 | sed "s/nrj_st1_e1_DH_protein_${RUN}/nrj_st1_e1_NA_ions_${RUN}/"  > tmp22
	mv tmp22 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_NA_ions_e1.out${RUN}
	
	# CL ions contribution
	cat  ${INPUT}.inp | sed "s/nox5st1_counterions_na.psf/nox5st1_counterions_cl.psf/" > tmp23
       	cat tmp23 | sed "s/nrj_st1_e1_NA_ions_${RUN}/nrj_st1_e1_CL_ions_${RUN}/"  > tmp24
	mv tmp24 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_CL_ions_e1.out${RUN}

	# Water ions contribution
	cat  ${INPUT}.inp | sed "s/nox5st1_counterions_cl.psf/nox5st1_water.psf/" > tmp25
       	cat tmp25 | sed "s/nrj_st1_e1_CL_ions_${RUN}/nrj_st1_e1_water_${RUN}/"  > tmp26
	mv tmp26 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_WATER_e1.out${RUN}
        
	# Propionate contribution
        cat  ${INPUT}.inp | sed "s/nox5st1_water.psf/nox5st1_propionate.psf/" > tmp27
        cat tmp27 | sed "s/nrj_st1_e1_water_${RUN}/nrj_st1_e1_propionate_${RUN}/"  > tmp28
        mv tmp28 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_PROP_e1.out${RUN}

	# Total energy of the system 
        cat  ${INPUT}.inp | sed "s/nox5st1_propionate.psf/nox5st1.psf/" > tmp29 
	cat tmp29 | sed "s/nrj_st1_e1_propionate_${RUN}/nrj_st1_e1_all_${RUN}/"  > tmp30
	mv tmp30 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_all_e1.out${RUN}

	rm -r tmp*

outputfile=${INPUT}_all_e1.out${RUN}

# Count the number of lines
nline=$( cat $outputfile | grep 'ENERGY:   ' | wc -l  | awk '{print $1}' )


for ((I=1;I<=$nline;I+=1)) ; do    # Loop over the snapshots of run $RUN and store energy in etot
    outputfile=${INPUT}_all_e1.out${RUN}
    etot[1]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_heme1_2_e1.out${RUN}
    etot[2]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )  
    outputfile=${INPUT}_heme1_e1.out${RUN}
    etot[3]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_heme2_e1.out${RUN}
    etot[4]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_fad_e1.out${RUN}
    etot[5]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_POPE_upper_e1.out${RUN}
    etot[6]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_POPG_upper_e1.out${RUN}
    etot[7]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_POPE_lower_e1.out${RUN}
    etot[8]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_POPG_lower_e1.out${RUN}
    etot[9]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_TM_protein_e1.out${RUN}
    etot[10]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_DH_protein_e1.out${RUN}
    etot[11]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_NA_ions_e1.out${RUN}
    etot[12]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_CL_ions_e1.out${RUN}
    etot[13]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_WATER_e1.out${RUN}
    etot[14]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_PROP_e1.out${RUN}
    etot[15]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )


    for J in `seq 1 15 ` ; do
      if [ $J -eq 1 ] ; then
        echo  " $RUN ${etot[$J]} "|awk '{printf("%s",$0)}'  >> ${INPUT}_e1_${START}_${END}.dat
      else
        echo  " ${etot[$J]} "|awk '{printf("%s",$0)}'  >> ${INPUT}_e1_${START}_${END}.dat
      fi
    done
    echo "" >> ${INPUT}_e1_${START}_${END}.dat
 done


        # Now we do the same for state 2

        cp ../${INPUT}.inp .
        cat  ${INPUT}.inp | sed "s/PSFPSFPSF/nox5st2_heme1_2.psf/" > tmp1
       	cat tmp1 | sed "s/_NUM/_${RUN}/" > tmp2
        cat tmp2 | sed "s/XXXXXX/nrj_st1_e2_heme1_2_${RUN}/"   > tmp3
	mv tmp3 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_heme1_2_e2.out${RUN}
 
        cat  ${INPUT}.inp | sed "s/nox5st2_heme1_2.psf/nox5st2_heme1.psf/" > tmp3
       	cat tmp3 | sed "s/nrj_st1_e2_heme1_2_${RUN}/nrj_st1_e2_heme1_${RUN}/"  > tmp4
	mv tmp4 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_heme1_e2.out${RUN}

        cat  ${INPUT}.inp | sed "s/nox5st2_heme1.psf/nox5st2_heme2.psf/" > tmp5
	cat tmp5 | sed "s/nrj_st1_e2_heme1_${RUN}/nrj_st1_e2_heme2_${RUN}/"  > tmp6
	mv tmp6 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_heme2_e2.out${RUN} 

	# Fad contribution 
        cat  ${INPUT}.inp | sed "s/nox5st2_heme2.psf/nox5st2_fad.psf/" > tmp7
       	cat tmp7 | sed "s/nrj_st1_e2_heme2_${RUN}/nrj_st1_e2_fad_${RUN}/"  > tmp8
	mv tmp8 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_fad_e2.out${RUN}
	
	# Upper leaflet POPE contribution
        cat  ${INPUT}.inp | sed "s/nox5st2_fad.psf/nox5st2_lipid_pope_upper_leaflet.psf/" > tmp9
       	cat tmp9 | sed "s/nrj_st1_e2_fad_${RUN}/nrj_st1_e2_POPE_upper_${RUN}/"  > tmp10
	mv tmp10 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_POPE_upper_e2.out${RUN}
        
	# Upper leaflet POPG contribution
	cat  ${INPUT}.inp | sed "s/nox5st2_lipid_pope_upper_leaflet.psf/nox5st2_lipid_popg_upper_leaflet.psf/" > tmp11
       	cat tmp11 | sed "s/nrj_st1_e2_POPE_upper_${RUN}/nrj_st1_e2_POPG_upper_${RUN}/"  > tmp12
	mv tmp12 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_POPG_upper_e2.out${RUN}
        
	# Lower leaflet POPE contribution
	cat  ${INPUT}.inp | sed "s/nox5st2_lipid_popg_upper_leaflet.psf/nox5st2_lipid_pope_lower_leaflet.psf/" > tmp13
       	cat tmp13 | sed "s/nrj_st1_e2_POPG_upper_${RUN}/nrj_st1_e2_POPE_lower_${RUN}/"  > tmp14
	mv tmp14 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_POPE_lower_e2.out${RUN}
	
	# Lower leaflet POPG contribution
	cat  ${INPUT}.inp | sed "s/nox5st2_lipid_pope_lower_leaflet.psf/nox5st2_lipid_popg_lower_leaflet.psf/" > tmp15
       	cat tmp15 | sed "s/nrj_st1_e2_POPE_lower_${RUN}/nrj_st1_e2_POPG_lower_${RUN}/"  > tmp16
	mv tmp16 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_POPG_lower_e2.out${RUN}
	
	# TM protein contribution
	cat  ${INPUT}.inp | sed "s/nox5st2_lipid_popg_lower_leaflet.psf/nox5st2_protein_tm_part.psf/" > tmp17
       	cat tmp17 | sed "s/nrj_st1_e2_POPE_lower_${RUN}/nrj_st1_e2_TM_protein_${RUN}/"  > tmp18
	mv tmp18 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_TM_protein_e2.out${RUN}
	
	# DH protein contribution
	cat  ${INPUT}.inp | sed "s/nox5st2_protein_tm_part.psf/nox5st2_protein_dh_part.psf/" > tmp19
       	cat tmp19 | sed "s/nrj_st1_e2_TM_protein_${RUN}/nrj_st1_e2_DH_protein_${RUN}/"  > tmp20
	mv tmp20 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_DH_protein_e2.out${RUN}
	
	# NA ions contribution
	cat  ${INPUT}.inp | sed "s/nox5st2_protein_dh_part.psf/nox5st2_counterions_na.psf/" > tmp21
       	cat tmp21 | sed "s/nrj_st1_e2_DH_protein_${RUN}/nrj_st1_e2_NA_ions_${RUN}/"  > tmp22
	mv tmp22 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_NA_ions_e2.out${RUN}
	
	# CL ions contribution
	cat  ${INPUT}.inp | sed "s/nox5st2_counterions_na.psf/nox5st2_counterions_cl.psf/" > tmp23
       	cat tmp23 | sed "s/nrj_st1_e2_NA_ions_${RUN}/nrj_st1_e2_CL_ions_${RUN}/"  > tmp24
	mv tmp24 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_CL_ions_e2.out${RUN}

	# Water ions contribution
	cat  ${INPUT}.inp | sed "s/nox5st2_counterions_cl.psf/nox5st2_water.psf/" > tmp25
       	cat tmp25 | sed "s/nrj_st1_e2_CL_ions_${RUN}/nrj_st1_e2_water_${RUN}/"  > tmp26
	mv tmp26 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_WATER_e2.out${RUN}

        # Propionate contribution
        cat  ${INPUT}.inp | sed "s/nox5st2_water.psf/nox5st2_propionate.psf/" > tmp27
        cat tmp27 | sed "s/nrj_st1_e2_water_${RUN}/nrj_st1_e2_propionate_${RUN}/"  > tmp28
        mv tmp28 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_PROP_e2.out${RUN}

        # Total energy of the system 
        cat  ${INPUT}.inp | sed "s/nox5st2_propionate.psf/nox5st2.psf/" > tmp29
        cat tmp29 | sed "s/nrj_st1_e2_propionate_${RUN}/nrj_st1_e2_all_${RUN}/"  > tmp30
        mv tmp30 ${INPUT}.inp
        namd2 ${INPUT}.inp > ${INPUT}_all_e2.out${RUN}


        rm -r tmp*

outputfile=${INPUT}_all_e2.out${RUN}

# Count the number of lines
nline=$( cat $outputfile | grep 'ENERGY:   ' | wc -l  | awk '{print $1}' )


for ((I=1;I<=$nline;I+=1)) ; do
    outputfile=${INPUT}_all_e2.out${RUN}
    etot[1]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_heme1_2_e2.out${RUN}
    etot[2]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_heme1_e2.out${RUN}
    etot[3]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_heme2_e2.out${RUN}
    etot[4]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_fad_e2.out${RUN}
    etot[5]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_POPE_upper_e2.out${RUN}
    etot[6]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_POPG_upper_e2.out${RUN}
    etot[7]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_POPE_lower_e2.out${RUN}
    etot[8]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_POPG_lower_e2.out${RUN}
    etot[9]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_TM_protein_e2.out${RUN}
    etot[10]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_DH_protein_e2.out${RUN}
    etot[11]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_NA_ions_e2.out${RUN}
    etot[12]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_CL_ions_e2.out${RUN}
    etot[13]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_WATER_e2.out${RUN}
    etot[14]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )
    outputfile=${INPUT}_PROP_e2.out${RUN}
    etot[15]=$(cat $outputfile | grep 'ENERGY:   ' | head -n $I | tail -n 1  |  awk '{print $14}' )

for J in `seq 1 15 ` ; do
      if [ $J -eq 1 ] ; then
        echo  " $RUN ${etot[$J]} "|awk '{printf("%s",$0)}'  >> ${INPUT}_e2_${START}_${END}.dat
      else
        echo  " ${etot[$J]} "|awk '{printf("%s",$0)}'  >> ${INPUT}_e2_${START}_${END}.dat
      fi
    done
    echo "" >> ${INPUT}_e2_${START}_${END}.dat
 done

done 

