import classifier.LogisticRegression
import util.Random

@main def hello(): Unit =
  println("Logistic regression !")

  // Generate synthetic training data
  val generator = Random(0)
  val X: Array[Array[Double]] = Array.ofDim(200, 2)
  val y: Array[Double] = Array.ofDim(200)
  for i <- 0 to 99 do 
    X(i)(0) = generator.nextGaussian()
    X(i)(1) = generator.nextGaussian()
    y(i) = 0
  for i <- 100 to 199 do 
    X(i)(0) = generator.nextGaussian() + 5
    X(i)(1) = generator.nextGaussian() + 5
    y(i) = 1

  // Generate synthetic validation data
  val Xval: Array[Array[Double]] = Array.ofDim(10, 2)
  val yval: Array[Double] = Array.ofDim(10)
  for i <- 0 to 4 do 
    Xval(i)(0) = generator.nextGaussian()
    Xval(i)(1) = generator.nextGaussian()
    yval(i) = 0
  for i <- 5 to 9 do 
    Xval(i)(0) = generator.nextGaussian() + 5
    Xval(i)(1) = generator.nextGaussian() + 5
    yval(i) = 1  
  
  // Instantiate the model
  val model = LogisticRegression()

  // Training of the model
  model.train(X, y, lr=0.1, maxIter = 200)

  // Prediction on validation set
  printf("yval: ")
  println(yval.toList)
  printf("predicted yval: ")
  println(model.predict(Xval).toList)
