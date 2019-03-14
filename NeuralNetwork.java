import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

// Neural Network
public class NeuralNetwork {

  Node[][] nodes;
  AdjNode[][] adjNodes;
  int layers;
  double learningRate = 0.1;

  // network[] = {8,8,8} -> Neural network with 3 layers each with 8 neurons/nodes
  NeuralNetwork(int[] network) {
    this.layers = network.length;
    this.nodes = new Node[layers][];
    this.adjNodes = new AdjNode[layers][];
    // Create structure
    for (int layer = 0; layer < this.layers; layer++) {
      this.nodes[layer] = new Node[network[layer]];
      this.adjNodes[layer] = new AdjNode[network[layer]];

    }

    // Initialize nodes
    Random rand = new Random();
    for (int i = 0; i < this.nodes.length; i++) {
      for (int j = 0; j < this.nodes[i].length; j++) {
        this.nodes[i][j] = new Node(rand, i, j, 0, (i > 0) ? network[i-1] : -1);
        this.adjNodes[i][j] = new AdjNode(i, j, (i > 0) ? network[i-1] : -1);
      }
    }
    
  }
  
  
  // Load in external neural network data (weights & biases)
  void loadData(double[] data) {
    int index = 0;
    for (int layer = 1; layer < this.layers; layer++) {
      for (int j = 0; j < this.nodes[layer].length; j++) {
        Node currNode = this.nodes[layer][j];
        for (int k = 0; k < currNode.weights.length; k++) {
          currNode.weights[k] = data[index];
          index += 1;
        }
        currNode.bias = data[index];
        index += 1;
      }
    }
  }
  
  // Train this neural network with a training input
  void train(TrainingInput tr) {
    double output[] = forwardPropagate(tr.input);
    
    // Find the loss
    double[] loss = new double[output.length];
    for (int i = 0; i < loss.length; i++) {
      loss[i] = 2 * (output[i] - tr.expected[i]);
    }
    
    // Backpropogate using SGD to adjust weights & biases
    this.nodes = this.backPropagate(loss).nodes;
  }


  // Train this neural network with a list of training inputs
  // Average all the desired changes to the weights & biases, and then apply to the network
  void trainBatch(ArrayList<TrainingInput> tr) {
    // Train whole batch, update adjNode[][]
    for (TrainingInput trainingInput : tr) {
      train(trainingInput);
    }
    
    int count = tr.size();
    
    // Update all weights and biases using adjNode[][]
    for (int layer = 1; layer < this.layers; layer++) {
      for (int j = 0; j < this.nodes[layer].length; j++) {
        Node currNode = this.nodes[layer][j];
        AdjNode currAdjNode = this.adjNodes[layer][j];
        currNode.bias = currNode.bias - (currAdjNode.deltaBias/count);
        for (int k = 0; k < currNode.weights.length; k++) {
          currNode.weights[k] = currNode.weights[k] - (currAdjNode.deltaWeights[k]/count);
        }
      }
    }
    
    // Reset adjNode[][]
    for (int layer = 1; layer < this.layers; layer++) {
      for (int j = 0; j < this.nodes[layer].length; j++) {
        AdjNode currAdjNode = this.adjNodes[layer][j];
        currAdjNode.deltaBias = 0;
        double[] newWeights = new double[currAdjNode.deltaWeights.length];
        for (int k = 0; k < currAdjNode.deltaWeights.length; k++) {
          newWeights[k] = 0;
        }
        currAdjNode.deltaWeights = newWeights;
      }
    }
    
  }
  
  // After training is over, validate with training input data to verify how "good" the network is. 
  // Returns the loss. 
  double validate(ArrayList<TrainingInput> tr) {
    double totalLoss = 0;
    for (TrainingInput trainingInput : tr) {
      double[] output = this.forwardPropagate(trainingInput.input);
      for (int i = 0; i < output.length; i++) {
        totalLoss += Math.pow((output[i]-trainingInput.expected[i]), 2);
      }
    }
    return totalLoss;
  }
  

  // Given an input, computes the output of the neural network
  double[] forwardPropagate(double[] input) {
   // For each layer:
      //   Set first row of nodes's a value equal to training input vector's given
      // For all other layers:
      //   Make a vector[] of a's from the previous layer's nodes[]
      //   Make a vector[] of biases from this layer's nodes[]
      //   Make a matrix[][] of weights from this layer's nodes[] with each node's weights[] being a row in the matrix
      //   Set a's in this layer's nodes to activationFunc(vectAdd(matVectMult(weightsMat, aVect), bVect))
      // Retrieve the final layer's a values as output[]
  
      // For each layer:
      for (int layer = 0; layer < this.layers; layer++) {
        //   Set first row of nodes's a value equal to training input vector's given
        if (layer == 0) {
          for (int i = 0; i < this.nodes[layer].length; i++) {
            this.nodes[layer][i] = new Node(this.nodes[layer][i], input[i], 0);
          }
        // For all other layers:
        } else {
          double[] aVect = new double[this.nodes[layer - 1].length];
          double[] bVect = new double[this.nodes[layer].length];
          double[][] weightsMat = new double[this.nodes[layer].length][this.nodes[layer - 1].length];
          
          // Make a vector of a's from the previous layer's nodes[]
          for (int i = 0; i < this.nodes[layer - 1].length; i++) {
            aVect[i] = this.nodes[layer - 1][i].a;
          }
          
          // Make a vector of biases from this layer's nodes[]
          // Make a matrix[][] of weights from this layer's nodes[] 
          // with each node's weights[] being a row in the matrix
          for (int i = 0; i < this.nodes[layer].length; i++) {
            bVect[i] = this.nodes[layer][i].bias;
            weightsMat[i] = this.nodes[layer][i].weights;
          } 
  
          // Apply activation function to (Wa + b)
          double[] zVals = Utils.vectAdd(Utils.matVectMult(weightsMat, aVect), bVect);
          double[] aVals = new double[zVals.length];
          for (int i = 0; i < aVals.length; i++) {
            aVals[i] = Utils.activationFunc(zVals[i]);
          }
          
          // Set a's in this layer's nodes to the activFunc(Wa + b)
          for (int i = 0; i < this.nodes[layer].length; i++) {
            this.nodes[layer][i] = new Node(this.nodes[layer][i], aVals[i], zVals[i]);
          }
  
        }
      }
      // Retrieve the final layer's a values as output[]
      double[] output = new double[this.nodes[this.layers-1].length];
      for (int i = 0; i < output.length; i++) {
        output[i] = this.nodes[this.layers-1][i].a;
      }
      
    //  Utils.printArr(output, "Output"); //output works
  
      return output;
    }


  // Given a loss function, back propagate using stochastic gradient descent (SGD) to 
  // adjust weights & biases accordingly. 
  NeuralNetwork backPropagate(double[] loss) {
    // For each layer (start from end):
    //   For each node(j) in layer:
    //     Node's bias = loss[j] * g'(z(j,L)), 
    //     else Summation[m=0, n-1](loss[m] * g'(z(m,L+1)) * w(mj,L+1)) * g'(z(j,L))
    //     For each weight(jk,L) in node: 
    //       if (currlayer == this.layers): 
    //         weight(jk,L) = loss[j] * g'(z(j,L)) * a(k, L-1)
    //       else:
    //         weight(jk,L) = 
    //         Summation[m=0, n-1](loss[m] * g'(z(m,L+1)) * w(mj,L+1)) * g'(z(j,L)) * a(k, L-1)
    //     
    
    // For each layer (start from end):
    for (int layer = this.layers-1; layer >= 0; layer--) {
      // For each node(j) in layer: 
      for (int j = 0; j < this.nodes[layer].length; j++) {
        Node currNode = this.nodes[layer][j];

        // If we are in the input layer, we are done and return this neural network
        if (layer == 0) {
          return this;
        }
        // BIAS ADJUSTMENT
        // Output layer
        if (layer == this.layers-1) {
          // Node's bias = loss[j] * g'(z(j,L)), 
          double adjBias = (loss[j] * Utils.activationFuncDer(currNode.z)) * this.learningRate;
  //      this.nodes[layer][j].bias = this.nodes[layer][j].bias - adjBias;
          this.adjNodes[layer][j].deltaBias += adjBias;
        } else {
          // Node's bias = Summation[m=0, n-1](loss[m] * g'(z(m,L+1)) * w(mj,L+1)) * g'(z(j,L))
          double adjBias = (Utils.activationFuncDer(currNode.z) * this.recur(layer+1, j, loss)) * this.learningRate;
          this.adjNodes[layer][j].deltaBias += adjBias;
  //      this.nodes[layer][j].bias = this.nodes[layer][j].bias - adjBias;
        }

        // WEIGHTS ADJUSTMENT

        // Output layer
        if (layer == this.layers - 1) {
          for (int k = 0; k < currNode.weights.length; k++) {
            // Node's weight(jk, L) = loss[j] * g'(z(j,L)) * a(k, L-1)
            double adjWeight = (loss[j] * Utils.activationFuncDer(currNode.z) * this.nodes[layer - 1][k].a) * this.learningRate;
            this.adjNodes[layer][j].deltaWeights[k] += adjWeight;
   //       this.nodes[layer][j].weights[k] = this.nodes[layer][j].weights[k] - adjWeight;
          }
        } else {
          for (int k = 0; k < currNode.weights.length; k++) {
            // weight(jk,L) =
            // Summation[m=0, n-1](loss[m] * g'(z(m,L+1)) * w(mj,L+1)) * g'(z(j,L)) * a(k, L-1)
            double adjWeight = (Utils.activationFuncDer(currNode.z) * this.nodes[layer - 1][k].a * this.recur(layer+1, j, loss)) * this.learningRate;
            this.adjNodes[layer][j].deltaWeights[k] += adjWeight;
   //       this.nodes[layer][j].weights[k] = this.nodes[layer][j].weights[k] - adjWeight;
          }
        }
      }
    }
    return this;
    
  }
  
  

  double recur(int layer, int j, double[] loss) {
    double ans = 0;
    if (layer != this.layers-1) {
      for (int m = 0; m < this.nodes[layer].length; m++) {
        ans += this.recur(layer + 1, m, loss) * Utils.activationFuncDer(this.nodes[layer][m].z)
            * this.nodes[layer][m].weights[j];
      }
    } else if (layer == this.layers-1) {
      for (int m = 0; m < this.nodes[layer].length; m++) {
        ans += loss[m] * Utils.activationFuncDer(this.nodes[layer][m].z)
            * this.nodes[layer][m].weights[j];
      }
    }
    return ans;
  }
}

// Node to keep track of how much we want to shift a certain node
class AdjNode {
  int layer; //index: row
  int n; //index: column
  double deltaBias;
  double[] deltaWeights;
  
  AdjNode(int layer, int n, int numOfWeights) {
    this.layer = layer;
    this.n = n;
    this.deltaBias = 0;
    if (numOfWeights != -1) {
      this.deltaWeights = new double[numOfWeights];
      for (int i = 0; i < this.deltaWeights.length; i++) {
        this.deltaWeights[i] = 0;
      }
    }
  }
  
  AdjNode(int layer, int n, double deltaBias, double[] deltaWeights) {
    this.layer = layer;
    this.n = n;
    this.deltaBias = deltaBias;
    this.deltaWeights = deltaWeights;
  }
  
  AdjNode(AdjNode an) {
    this(an.layer, an.n, an.deltaBias, an.deltaWeights);
  }
  
  AdjNode(AdjNode adjNode, double deltaBias) {
    this(adjNode);
    this.deltaBias = deltaBias;
  }
  
  AdjNode(AdjNode adjNode, double[] deltaWeights) {
    this(adjNode);
    this.deltaWeights = deltaWeights;
  }

}


// Represents a neuron in a neural network
class Node {
  int layer; //index: row
  int n; //index: column
  double a;
  double bias;
  double[] weights;
  double z;

  Node(Random rand, int layer, int n, double a, int numOfWeights) {
    this.layer = layer;
    this.n = n;
    this.a = a;
   // this.bias = rand.nextGaussian();
    this.bias = 1;
    if (numOfWeights != -1) {
      this.weights = new double[numOfWeights];
      for (int i = 0; i < this.weights.length; i++) {
        // Xavier Initialization
        // if using relu activation, use Math.sqrt(2/weights.length)
        this.weights[i] = rand.nextGaussian() * Math.sqrt(1.0/weights.length);
      }
    }
    z = 0;
  }

  Node(int layer, int n, double a, double bias, double[] weights) {
    this.layer = layer;
    this.n = n;
    this.a = a;
    this.bias = bias;
    this.weights = weights;
  }

  Node(Node n) {
    this(n.layer, n.n, n.a, n.bias, n.weights);
  }

  Node(Node node, double a, double z) {
    this(node);
    this.a = a;
    this.z = z;
  }
  
  Node(Node node, double b) {
    this(node);
    this.bias = b;
  }
  
  Node(Node node, double[] weights) {
    this(node);
    this.weights = weights;
  }

}


// A training input with an input and an expected output used to train the neural network.
class TrainingInput {

  double[] input;
  double[] expected;

  TrainingInput(double[] input, double[] expected) {
    this.input = input;
    this.expected = expected;
  }

}


class Utils {
  // Multiply a matrix by a vector (Mv = v')
  // Returns the resultant vector
  static double[] matVectMult(double[][] mat, double[] vect) {
    if (mat[0].length != vect.length) {
      throw new IllegalArgumentException("Cannot multiply this matrix by this vector");
    }
    double[] outputV = new double[mat.length];
    for (int i = 0; i < mat.length; i++) {
      double sum = 0;
      for (int j = 0; j < mat[i].length; j++) {
        sum += mat[i][j] * vect[j];
      }
      outputV[i] = sum;
    }
    return outputV;
  }

  // Adds to vectors, returns the resultant vector
  static double[] vectAdd(double[] vect1, double[] vect2) {
    if (vect1.length != vect2.length) {
      throw new IllegalArgumentException("Vectors in vectAdd of different length!");
    }
    double[] outputV = new double[vect1.length];
    for (int i = 0; i < vect1.length; i++) {
      outputV[i] = vect1[i] + vect2[i];
    }
    return outputV;
  }
  
  // Activation function
  static double activationFunc(double x) {
    return sigmoidFunc(x);
    //return reluFunc(x);
  }
  
  // Derivative of the activation function
  static double activationFuncDer(double x) {
    return sigmoidFuncDer(x);
    //return reluFunc(x);
  }

  // Sigmoid Function
  private static double sigmoidFunc(double x) {
    return 1 / (1 + Math.exp(-x));
  }

  @SuppressWarnings("unused")
  // ReLU Function
  private static double reluFunc(double x) {
    return Math.max(0, x);
  }

  // Derivative of Sigmoid
  private static double sigmoidFuncDer(double x) {
    return sigmoidFunc(x) * (1 - sigmoidFunc(x));
  }

  @SuppressWarnings("unused")
  // Derivative of ReLU
  private static double reluFuncDer(double x) {
    if (x > 0) {
      return 1;
    } else {
      return 0;
    }
  }
  
  // Print the given array
  static void printArr(double[] arr, String name) {
    System.out.println(name + ": ");

    for (int i = 0; i < arr.length; i++) {
      System.out.println("Element " + i + ": " + arr[i]);
    }
  }

  // Print the neural network in a visual manner
  public static void printNeuralNetwork2(NeuralNetwork nn) {
    Node[][] n = nn.nodes;
    for (int layer = 0; layer < n.length; layer++) {
      System.out.println("Layer " + layer + ": ");
      for (int j = 0; j < n[layer].length; j++) {
        if (layer == 0) {
          System.out.println("  Node " + j);
        } else {
          Node currNode = n[layer][j];
          System.out.println("  Node " + j);
          for (int k = 0; k < currNode.weights.length; k++) {
            System.out.println("    Weight " + k + ": " + currNode.weights[k]);
          }
          System.out.println("    Bias: " + currNode.bias);
        }
      }
    }
  }
  
  // Print the weights and biases of a neural network
  public static void printNeuralNetwork(NeuralNetwork nn) {
    String output = "{";
    Node[][] n = nn.nodes;
    for (int layer = 1; layer < n.length; layer++) {
      for (int j = 0; j < n[layer].length; j++) {
          Node currNode = n[layer][j];
          for (int k = 0; k < currNode.weights.length; k++) {
            output += currNode.weights[k] + ", ";
          }
          output += currNode.bias + ", ";
      }
    }
    System.out.println(output.substring(0, output.length()-2) + "}");
  }

  // Take user input, so the user can test inputs of their choosing
  public static void takeUserInput(NeuralNetwork nn) {
    System.out.println("User input activated, type stop to exit.");
    boolean done = false;
    Scanner sc = new Scanner(System.in);
    while (!done) {
      String str = sc.nextLine();
      if (str.equals("stop")) {
        sc.close();
        done = true;
      } else {
        String[] splitStr = str.split(",");
        double[] input = new double[splitStr.length];
        for (int i = 0; i < splitStr.length; i++) {
          input[i] = Double.parseDouble(splitStr[i]);
        }
        printArr(nn.forwardPropagate(input), "Output"); 
      }
    }
    
  }
  
}