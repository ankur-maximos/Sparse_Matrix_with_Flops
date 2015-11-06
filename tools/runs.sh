dataDir=/home/niuq/data/matrix_input
inputs="cant.mtx  consph.mtx  mc2depi.mtx filter3D.mtx cop20k_A.mtx  mac_econ_fwd500.mtx  mc2depi.mtx  pdb1HYS.mtx  pwtk.mtx rma10.mtx  scircuit.mtx  webbase-1M.mtx shipsec1.mtx 2cubes_sphere.mtx cage12.mtx hood.mtx m133-b3.mtx majorbasis.mtx mario002.mtx mono_500Hz.mtx offshore.mtx patents_main.mtx poisson3Da.mtx qcd5_4.mtx com-amazon.ungraph.mtx email-Enron.mtx roadNet-CA.mtx  twitter_combined.mtx web-NotreDame.mtx BioGRID_unweighted_graph.mtx  com-dblp.ungraph.mtx  com-youtube.ungraph.mtx facebook_combined.mtx  web-BerkStan.mtx WIPHI_graph.mtx cit-HepPh.mtx DIP_unweighted_graph.mtx  loc-gowalla_edges.mtx  web-Google.mtx"
stride=512
bin=bin_analysis.x
for input in $inputs; do
  echo $dataDir/$input
    command="./$bin --stride=$stride --input=$dataDir/$input"
    echo $command
    ./$bin --stride=$stride --input=$dataDir/$input -m 30 
    #$command
done
