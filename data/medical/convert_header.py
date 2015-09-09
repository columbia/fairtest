features=['CharlsonIndexI_ave','CharlsonIndexI_max','CharlsonIndexI_min','CharlsonIndexI_range','CharlsonIndexI_stdev','ClaimsTruncated','LOS_TOT_KNOWN','LOS_TOT_SUPRESSED','LOS_TOT_UNKNOWN','LOS_ave','LOS_max','LOS_min','LOS_stdev','PayDelay_ave','PayDelay_max','PayDelay_min','PayDelay_stdev','drugCount_ave','drugCount_max','drugCount_min','drugNull','drugcount_months','dsfs_ave','dsfs_max','dsfs_min','dsfs_range','dsfs_stdev','labCount_ave','labCount_max','labCount_min','labNull','labcount_months','no_Claims','no_PCPs','no_PlaceSvcs','no_PrimaryConditionGroups','no_ProcedureGroups','no_Providers','no_Specialities','no_Vendors']

rep_features = ['Average Charlson Index','Max Charlson Index', 'Min Charlson Index', 'Range Charlson Index', 'Stdev Charlson Index', 'Suppressed Claims', 'Known Length Of Stay','Supressed Length Of Stay','Unknown Length Of Stay', 'Average Length Of Stay', 'Max Length Of Stay', 'Min Length Of Stay', 'Stdev Length Of Stay', 'Average Payment Delay', 'Max Payment Delay', 'Min Payment Delay', 'Stdev Payment Delay', 'Average Drug Count', 'Max Drug Count', 'Min Drug Count', 'No Drug Count', 'Drug Months', 'Average Days Since First Service', 'Max Days Since First Service', 'Min Days Since First Service', 'Range Days Since First Service', 'Stdev Days Since First Service', 'Average Lab Count', 'Max Lab Count', 'Min Lab Count', 'No Lab Count', 'Lab Months', 'Number Of Claims', 'Number of Primary Care Physicians', 'Number of Places Where Treated', 'Number of Primary Condition Groups', 'Number of Procedure Groups', 'Number of Providers', 'Number of Specialities', 'Number of Vendors']

assert len(features) == len(rep_features)

groups=[]

for i in range(1, 46+1):
    groups.append('pcg{}'.format(i))

for i in range(1, 13+1):
    groups.append('sp{}'.format(i))
    
for i in range(1, 18+1):
    groups.append('pg{}'.format(i))
    
for i in range(1, 9+1):
    groups.append('ps{}'.format(i))

rep_pcgs=['MISC2','METABOLIC','ARTHROPATHIES','NEUROLOGICAL OTHER','RESPIRATORY','MISC CARDIAC','SKIN AND AUTOIMMUNE DISORDERS','GASTROINTESTINAL BLEEDING', 'INFECTION', 'TRAUMA', 'CARDIAC', 'RENAL OTHER', 'CHEST PAIN', 'MISC3', 'INGESTIONS AND BENIGN TUMORS', 'URINARY TRACT INFECTIONS', 'CHRONIC OBSTRUCTIVE PULMONARY DISORDER', 'GYNECOLOGY', 'CANCER B', 'FRACTURES AND DISLOCATIONS', 'ACUTE MYOCARDIAL INFARCTION', 'PREGNANCY', 'NON-­MALIGNANT HEMATOLOGIC', 'ATHEROSCLEROSIS ', 'SEIZURE', 'APPENDICITIS', 'CONGESTIVE HEART FAILURE', 'GYNECOLOGIC CANCERS', 'NULL', 'PNEUMONIA', 'CHRONIC RENAL FAILURE','GASTR. INFLAM. BOWEL DISEASE AND OBSTRUCTION', 'STROKE', 'CANCER A', 'FLUID AND ELECTROLYTE', 'MISC1', 'HIP FRACTURE', 'DIABETIC', 'PERICARDITIS', 'LIVER DISORDERS', 'CATASTROPHIC CONDITIONS', 'OVARIAN AND METASTATIC CANCER', 'PERINATAL PERIOD', 'PANCREATIC DISORDERS','ACUTE RENAL FAILURE', 'SEPSIS']
rep_pcgs = ['Condition '+pcg for pcg in rep_pcgs]
assert len(rep_pcgs) == 46

rep_sps=['Internal','Laboratory','General Practice','Surgery','Diagnostic Imaging','Emergency','Other','Pediatrics','Rehabilitation','Obstetrics and Gynecology','Anesthesiology','Pathology','NULL']
rep_sps = ['Speciality '+sp for sp in rep_sps]
assert len(rep_sps) == 13

rep_pgs=['EM','PL','MED','SCS','RAD','SDS','SIS','SMS','ANES','SGS','SEOA','SRS','SNS','SAS','SUS','NULL' ,'SMCD','SO']
rep_pgs = ['Treatment Code '+pg for pg in rep_pgs]
assert len(rep_pgs) == 18

rep_pss=['Office','Independent Lab','Urgent Care','Outpatient Hospital','Inpatient Hospital','Ambulance','Other','Home','NULL']
rep_pss = ['Treated at '+ps for ps in rep_pss]
assert len(rep_pss) == 9

tot_features = features + groups
tot_rep_features = rep_features + rep_pcgs + rep_sps + rep_pgs + rep_pss
assert len(tot_features) == len(tot_rep_features)

import sys
infile = sys.argv[1]

data = None
new_header = None

with open(infile) as inf:
    data = inf.readlines()
    header = data[0].rstrip().split(',')
    new_header = []
    
    for feature in header:
        try:
            idx = tot_features.index(feature)
            new_header.append(tot_rep_features[idx])
        except ValueError:
            print 'Feature {} Not Found!'.format(feature)
            new_header.append(feature)
            
with open(infile, 'w') as outf:
    outf.write(', '.join(new_header) + '\n')
    
    for line in data[1:]:
        outf.write(line)
    
    

