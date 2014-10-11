dataDir=/home/niuq/data/matrix_input
inputs="facebook_combined.txt email-Enron.txt com-dblp.ungraph.txt roadNet-CA.txt loc-gowalla_edges.txt com-youtube.ungraph.txt BioGRID_unweighted_graph.txt DIP_unweighted_graph.txt WIPHI_graph.txt com-amazon.ungraph.txt web-Google.txt"
stride=512
bin=analysis.x
for input in $inputs; do
  echo $dataDir/$input
    command="./$bin --stride=$stride --input=$dataDir/$input"
    echo $command
    ./$bin --stride=$stride --input=$dataDir/$input -m 30 
    #$command
done
