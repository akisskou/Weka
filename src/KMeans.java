import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.ClusterEvaluation;

public class KMeans {

	public static void main(String[] args) {
        // TODO code application logic here
		try{
            DataSource src = new DataSource("randomPoints.arff");
            Instances dataset = src.getDataSet();
            SimpleKMeans model = new SimpleKMeans();
            
            
            model.setPreserveInstancesOrder(true);
            model.setNumClusters(2);
            model.buildClusterer(dataset);
            System.out.println("DataPoints in each cluster:");
    		for (Instance instance : dataset) {
                System.out.println( 
                        instance.toString()+": "+
                        model.clusterInstance(instance));
            }
 
            ClusterEvaluation eval = new ClusterEvaluation();
            DataSource src1 = new DataSource("randomPoints.test.arff");
            Instances tdt = src1.getDataSet();
            eval.setClusterer(model);
            eval.evaluateClusterer(tdt);
            System.out.println(eval.clusterResultsToString());
            //System.out.println("# of clusters: " + eval.getNumClusters());
		}
        catch(Exception e)
        {
            System.out.println(e.getMessage());
        }
    }
}
