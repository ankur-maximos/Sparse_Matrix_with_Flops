dataDir=/home/niuq/data/matrix_input
inputs="com-amazon.ungraph.txt email-Enron.txt roadNet-CA.txt  twitter_combined.txt web-NotreDame.txt BioGRID_unweighted_graph.txt  com-dblp.ungraph.txt  com-youtube.ungraph.txt facebook_combined.txt  pwtk.txt web-BerkStan.txt WIPHI_graph.txt cit-HepPh.txt DIP_unweighted_graph.txt  loc-gowalla_edges.txt  web-Google.txt com-lj.ungraph.txt cant.mtx  consph.mtx  cop20k_A.mtx  mac_econ_fwd500.mtx  mc2depi.mtx  pdb1HYS.mtx  pwtk.mtx  rail4284.mtx  rail4284.tar.gz  rma10.mtx  scircuit.mtx  webbase-1M.mtx"
stride=512
bin=analysis.x
for input in $inputs; do
  echo $dataDir/$input
    command="./$bin --stride=$stride --input=$dataDir/$input"
    echo $command
    ./$bin --stride=$stride --input=$dataDir/$input -m 30 
    #$command
done
