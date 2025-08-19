## Copying and running a bash file on all the Pis
'''bash
while IFS= read -r h; do [ -z "$h" ] && continue; echo "== Copying and running on $h =="; scp -q systemprep_pi.sh pi@"$h":~ || { echo "scp failed for $h"; continue; }; ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new pi@"$h" 'bash ~/systemprep_pi.sh' || echo "ssh/script failed on $h"; done < hosts.txt
'''
