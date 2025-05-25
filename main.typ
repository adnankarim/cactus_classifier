#import "template.typ": *
#import "@preview/fletcher:0.5.2" as fletcher: diagram, node, edge
#import fletcher.shapes: house, hexagon, diamond

#import "@preview/wordometer:0.1.4": word-count, total-words

#let appendix(body) = {
  set heading(numbering: "A.1.", supplement: [Appendix])
  counter(heading).update(0)
  body
}

#show: project.with(
  title: "AML Challenge 1",
  subtitle: "Cactus Binary Classification",
  authors: (
    "PAN Qizhi ", "KARIM Adnan ",
  ),
  date: datetime.today().display("11. [month repr:long] [year]")
)
= Introduction
In this challenge, we tackle a binary classification task involving 32×32-pixel aerial images to determine the presence of a specific cactus species. Our approach divides the problem into two stages: feature extraction using convolutional neural networks (CNNs), followed by a binary classifier trained on the extracted features.

For feature extraction, we first adopt ResNet50, a widely used CNN with residual connections, as our benchmark model. While ResNet50 excels in many image recognition tasks, we hypothesize its deep architecture (50 layers) may struggle with the small input size, potentially losing fine-grained spatial details critical for identifying subtle cactus patterns.

To address this limitation, we test EfficientNet, a modern architecture designed for parameter efficiency and scalability. Its compound scaling mechanism balances depth, width, and resolution, making it better suited for low-resolution images like our 32×32 inputs.

To fairly compare the feature extraction capabilities of ResNet50 and EfficientNet, we keep the downstream classifier identical(sigmoid) for both models, ensuring performance differences reflect only the quality of the extracted features.



= Data Analysis and Preprocessing
For this task, we use all the images located in the ./train directory as our dataset. The dataset contains a total of 17,500 images, with 13,136 labeled as '1' (containing cactus) and 4,364 labeled as '0' (no cactus), indicating a class imbalance toward cactus-containing images.
== Analysis
We begin by visually inspecting sample images from both classes:



From the visual comparison, we observe that images labeled as containing cactus often exhibit:

- More linear textures (corresponding to the shape and structure of cactus)

- Greater presence of green hues, likely due to vegetation

These visual cues suggest that the key discriminative features for our neural network should include the ability to detect fine textures and color distribution patterns associated with vegetation.


== Preprocessing
The dataset was first split into 70% training, 15% validation, and 15% test subsets using stratified sampling to preserve class balance. Files were reorganized into subfolders by label (0 or 1) for compatibility with ImageFolder.

A ToTensor() transform was applied initially, and dataset-wide mean and standard deviation values were computed to support future normalization steps. The following statistics were calculated from the training set:

    Mean: [R_mean, G_mean, B_mean] (actual values printed in logs)

    Standard Deviation: [R_std, G_std, B_std]

These statistics can be used in normalization transforms in further experiments.
== Class Imbalance Handling

To address class imbalance in the dataset, we employed a weighted cross-entropy loss function. Class weights were computed based on the inverse frequency of each class in the training set, assigning higher weight to the minority class to ensure fairer learning.

These weights were converted to a tensor and passed to the `CrossEntropyLoss` criterion in PyTorch. If CUDA was enabled, the tensor was moved to the GPU. This ensures that the model does not become biased toward the majority class by penalizing errors on the minority class more heavily.

The class weights used were:

    Class 0 (majority): w₀ = 0.2494
    Class 1 (minority): w₁ = 0.7506

These values were applied during training to improve class-level performance and overall model generalization.

= Benchmark Model and performance

== Resnet50
We used a pretrained ResNet-50 model from torchvision.models as a base. All convolutional layers were frozen to leverage pretrained features. The final fully connected layers were replaced with a custom classifier head.
#figure(
  image("Resnet50_structure.png",width: 100%),
  caption: [Architecture of the ResNet-50 residual neural network#footnote[Copyright from https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f]]
)


``

This head was designed to introduce non-linearity while progressively reducing feature dimensionality to a binary output space.

== Performance
We trained the model with epoch = 500 and lr = 0.0005.
#figure(
  image("Resnet_loss.png",width: 75%),
  caption: [Training loss of Resnet50 with epoch = 500, lr = 0.0005]
)


#figure(
  caption: "Confusion Matrix",
  table(
    columns: 3,
    rows: 3,
    align: center,
    fill: none,
    stroke: 0.5pt,
    inset: 6pt,
    
    [] ,             [*Pred: 0*], [*Pred: 1*],
    [*True: 0*],[ 1924],  [           41],
    [*True: 1*],[   46],[            614],
  )
)

#figure(
  table(
    columns: 2,
    [AUC score],[95.7%],
    [Accuracy],[96.7%],
    [F1 score],[93.4%],
    [Precision score],[93.0%],
    [Recall score],[93.7%],
  ),
  caption: [Metrics]
)
Although ResNet50 performs well (96.7% accuracy, 93.4% F1 score), these results must be considered alongside class imbalance. F1, precision, and recall provide more meaningful insights than accuracy alone. The model performs reliably, but with some misclassifications




== Observation
The model required significant training time and epochs to stabilize, as shown in its loss curve. This slow convergence and moderate misclassification rate suggest that ResNet50 is not ideally suited for low-resolution image tasks under resource constraints.

To address these limitations, we explored more computationally efficient architectures. Among them, *EfficientNet* stands out for its ability to achieve high accuracy with significantly fewer parameters and FLOPs, making it a promising candidate for replacing ResNet50 in this context.
= Improved Method

== EfficientNet
To cope with the problems that the ResNet50 might have, we employed EfficientNet-B0, a lightweight CNN with approximately 5.3 million parameters and 0.39 billion FLOPs, which achieves higher accuracy than ResNet-50 (25.6M parameters, 4.1B FLOPs) while requiring significantly less computation.

Compared to ResNet-50, EfficientNet-B0 offers a ~5× reduction in parameters and ~10× reduction in FLOPs, while improving accuracy.

#figure(
  image("EfficientNet_para_table.png"),
  caption: [EfficientNet B0 structure in table]
)
== Performance
Here we still trained the model with epoch = 500 and lr = 0.0005.
#figure(
  image("Effi_loss.png",width: 75%),
  caption: [Training loss of EfficientNet with epoch = 500, lr = 0.0005]
)


#figure(
  caption: "Confusion Matrix",
  table(
    columns: 3,
    rows: 3,
    align: center,
    fill: none,
    stroke: 0.5pt,
    inset: 6pt,
    
    [] ,             [*Pred: 0*], [*Pred: 1*],
    [*True: 0*],[ 1961],  [           0],
    [*True: 1*],[   9],[            655],
  )
)

#figure(
  table(
    columns: 2,
    [AUC score],[99.8%],
    [Accuracy],[99.6%],
    [F1 score],[99.3%],
    [Precision score],[98.6%],
    [Recall score],[100%],
  ),
  caption: [Metrics]
)
== Comparison
We compare the performance of EfficientNet and ResNet based on their classification accuracy, evaluation metrics, and convergence behavior. The results clearly demonstrate the superior performance of EfficientNet in both predictive capability and training dynamics.

EfficientNet achieved an impressive accuracy of 99.6%, with an AUC score of 99.8%, F1 score of 99.3%, and precision/recall scores of 98.6% and 100%, respectively. The corresponding confusion matrix indicates that only 9 false negatives and 0 false positives occurred, highlighting its excellent ability to distinguish between classes.

In contrast, ResNet obtained a lower accuracy of 96.7%, with an AUC score of 95.7%, F1 score of 93.4%, precision of 93.0%, and recall of 93.7%. The confusion matrix shows more misclassifications, with 41 false positives and 46 false negatives, reflecting its relatively weaker performance on this task.

Furthermore, the training loss curve of EfficientNet was remarkably smooth and stable, converging rapidly to below 0.1 and remaining near zero throughout training. This suggests better generalization and optimization behavior compared to ResNet.
= Conclusion
The superior performance of EfficientNet over ResNet in this classification task can be largely attributed to its more refined architectural design for feature extraction. EfficientNet uses a compound scaling method that uniformly scales depth, width, and resolution, allowing the model to maintain a balanced capacity for capturing spatial and semantic features. Its depthwise separable convolutions and squeeze-and-excitation blocks further enhance its ability to extract discriminative features from images while keeping the model lightweight.

Compared to ResNet, which relies heavily on increasing depth through residual connections, EfficientNet achieves better feature representation with fewer parameters, leading to improved generalization. The early and consistent convergence of the loss curve highlights the model’s efficiency in learning robust features quickly and stably.

In summary, EfficientNet's design allows for deeper semantic understanding and more effective use of spatial hierarchies in the input data, making it a superior choice for image classification tasks where both accuracy and computational efficiency are essential.





/*

#pagebreak()
= Perceptron Model Revised 
From the last submission, we got two suggestions:

+ The training/validation/testing setup needs to be revised.
+ Can it happen that you get two ones from two classifiers? 

For comment 1, we examined our model and found that there's no problem in our testing set split, but we kept splitting the training set and validation set in the training session. We revised it and make sure all the sets are well split at the beginning.

For comment 2, initially, we adopted a “first come, first serve” strategy, but we have now moved to a more robust tie-break based on each classifier’s F1-score: whichever classifier (digit model) has a higher F1-score takes priority in claiming the sample. This better reflects the confidence of individual classifiers when conflicts arise.
#figure(
  image("revised perceptron model.png"),
  caption: [Revised combined perceptron model]
)

= Solution

== SAMME Implementation

To implement SAMME, we need to determine the weak learner, how to relate the sample weights to the weak learner and how to update the weak learner weights.
=== Brief Understanding
Multi-class AdaBoost (SAMME) extends AdaBoost to handle multiple classes by using a modified exponential loss function and combining multi-classifier weak learners trained iteratively, where each learner focuses on correcting the mistakes of its predecessors.


=== Weak Learner
We use our combined perceptron model as the weak learner. The combined perceptron model itself is a multi-classifier consists of 10 simple perceptrons as binary classifiers,  following one-vs-all strategy. 

The parameters related to it are the learning rate *alpha* and training *epoches*.
#pagebreak()
=== Sample Weights Implement
One of the keys to successfully implement the SAMMES is to implement the sample weights correctly to the weak learners. 

Sample weights affect the loss function, which further affect the weights and bias update rules in perceptron training. Here we implement them in update rules as:



$ w←w+α⋅omega_i⋅ y _ i⋅x_i $


$ b←b+α⋅omega_i⋅y_i $



Here, $omega_i$​ is the weight assigned to the i-th sample, and α is the learning rate. $w$ and $b$ are the weights and bias in perceptron model.

=== Weak Learner Weights Update
After completely trained the $t_("th")$ weak learner, we calculate its error as:

 $ "Error"_t=frac(sum_(i:y_i​ eq.not hat(y_(​i)) ​) omega_i, ​sum_(i=1)^n​omega_i) $​​
The contribution of the $t_("th")$ weak learner to the final model is determined by:

$ alpha_t = log((1-"Error"_t)/("Error"_t)) + log(n_"classes" - 1) $

The sample weights $omega_i$​ are updated to emphasize misclassified samples:
$ omega_i←omega_i exp( alpha_t · 1(y_i​ eq.not hat(y_(​i))^t)) $

where

$omega_i$ = Weight of the $i_("th")$ sample. 

$y_i$​ = True label of the $i_("th")$ sample. 
$hat(y_(​i))​$ = Predicted label of the $i_("th")$ sample.

$n_"classes"$​ = Total number of classes in the classification task. In our case, $n_"classes"$​ =10.

$1(y_i​ eq.not hat(y_(​i))^t)$ is an indicator function that equals 1 if the prediction is incorrect, and 0 otherwise.

$α_t$​ = Weight of the $i_("th")$ weak learner.

#figure(
  image("SAMMES illus.png",height: 28%),
  caption: [SAMMES implement illustration]
)
== Model Selection using K-Fold Cross Validation
=== K-Fold Cross Validation Implement
To determine the optimal hyperparameters for our multi-class AdaBoost (SAMME) model, we conducted a k-fold cross-validation process that incorporates all aspects of the pipeline—i.e., training T rounds of Perceptron-based boosting for each fold. This ensures the final hyperparameters we choose are robust to variations in our training data.

1. *Cross-Validation Setup*:
   - Define a small array of potential hyperparameters (e.g., `alpha ∈ {0.001, 0.005, 0.01}`)
   - Use k-fold (e.g., *k = 5*) to split data into train/validation folds. Train the Perceptron on *k-1* folds and measure accuracy on the remaining fold.

2. *Averaging and Selection*:
   - For each `(alpha, epochs, T)` combination, compute the average validation accuracy over the *k* folds.
   - Pick the combination with the highest average accuracy (or F1-score).

3. *Applying the Best Hyperparameters*:
   - Use the “best” `(alpha, epochs, T)` from cross-validation in the SAMME loop. This ensures each weak learner Perceptron is reasonably tuned.



   


=== Data Splitting and Hyperparameters

- *Train/Validation/Test Split* = 8:1:1
- *Hyperparameters*:
  - Number of rounds \( T \) in SAMME: [ 3, 5, 10 ]
  - Perceptron learning rate (`alpha`) : [ 0.001, 0.005, 0.01, 0.1]
  - Perceptron training epochs : [ 10, 20, 30, 50, 80, 100 ]

=== Results and Model Selection
To notice, for learning rate = 0.2, we only test it for epochs=[ 50, 80, 100 ], which shows exactly the same performance as learning rate = 0.1 and learning rate = 0.005 shows exactly the same result as 0.01.

From the plots below we can see that, the performances for almost  all the settings have obvious increase from epoch = 10 to epoch = 30. It means that for a quick, efficient and acceptable result, we could set the epoch to be 30. Epoch = 30 can be an elbow point for all other settings.

Increasing the number of weak learners T in SAMME training always improves the model performance. The monotonically increasing features of the model with T are also consistent with the design of the SAMME model. Trade off in this aspect is the model performance and the training cost.
#pagebreak()
 
#figure(
  image("F1_Kfold.png",width: 100%),
  caption: [F1 score in K-fold cross validation]
)

#figure(
  image("acc_Kfold.png",width: 100%),
  caption: [Accuracy in K-fold cross validation]
)
In terms of the highest performance, we notice that (10, 0.1, 100) and (10, 0.2, 100) have exactly the same result. Concerning the training cost, we won't test the performance when T>10 and epoch>100. *We choose the (10, 0.1, 100) as the best parameters for the final testing*, because higher learning rate might show higher risk for non-converging.

#pagebreak()
= Best Model and Analysis
We select the best model with parameters (T = 10, learning rate = 0.1, epochs = 100). 

The performance is shown below: 


#figure(
  table(
    table.header(
      [*Metric*], [*Value*],[*Metric*], [*Value*]
    ),
    columns: 4,
    "Validation Accuracy", "96.11%",
    "Testing Accuracy", "97.22%",
    "Validation F1-score", "0.96",
    "Testing F1-score", "0.97",
  ),
  caption: [Best SAMME model performance]
)
#figure(
  image("best_performance.png",width:100%),
  caption: [Best-parameter SAMME Model Results. Green areas mean the correct prediction and red areas mean the wrong prediction. The more counts, the deeper color. ]
)
#figure(
  image("F1_in_single_perceptron.png",width: 100%),
  caption: [F1 score for each single perceptrons in one weak learner]
)
This graph shows the f1 score of each simple perceptron inside a weak learner. Through this we can see how the sample weights change impacts the next round of learning. 

For weak leaner 0, we can see the simple perceptron 4 and simple perceptron 6 have low performance in classifying digit 4 and digit 6. In the next round, due to the sample weights change, this two simple perceptrons pay more attention to the training and get better performance.
#figure(
  image("weak_learner_performance.png",width: 100%),
  caption: [Performance of each weak learners]
)
This graph shows the error each weak learner makes during the SAMME training and the weights allocated to it. We can see that the errors and the weights are approximately inversely proportional. 


It's worth noticing that the weak learner 3 and weak learner 8 contribute most to the SAMME model. This good performance might attribute to all the single perceptrons they contain have good performance which can be examined from figure 6. This possible explanation works for weak learner 3 but is not quite suitable for weak learner 8. The specific reasons need further analysis.

= Discussion

== Observations on Performance

Our experiments demonstrated that increasing both the number of rounds \(T\) in SAMME and the Perceptron epochs leads to consistently improved accuracy and F1-scores. This aligns with the AdaBoost principle that boosting rounds help emphasize difficult examples, while additional training epochs refine each Perceptron’s decision boundary. However, these improvements come at the cost of longer training time, indicating a trade-off between computational resources and marginal gains in performance.

In particular, the model with \(\alpha = 0.1\), \(\text{epochs} = 100\), and \(T = 10\) came out as our best settings, achieving around 97% accuracy and a 0.97 F1-score on the test set. Interestingly, higher learning rates (e.g., 0.2) showed similar performance in some runs but might risk some instability in convergence. Lower rates, however, required more epochs or additional boosting rounds to attain equivalent performance.

== Weak Learners and Confusion Patterns

Our detailed analysis of single Perceptrons in each weak learner revealed that certain digits (like 4 and 6) initially was suffering from low F1-scores. Never the less, Some boosting rounds re-weighted these misclassified samples, allowing for later Perceptrons to focus on these problematic digit samples. This re-emphasis is key to AdaBoost’s success and can be observed in the metrics where weak learner performance gradually converges for these difficult classes.

The confusion matrix and error metrics for each round show that digits (e.g., digits 8 and 9) remain challenging even with the boosted re-weighting. While error rates were generally low, future improvements—such as feature engineering or more sophisticated tie-breaking strategies could further reduce these confusions.

== Practical Considerations

- *Model Complexity vs. Training Time*: While increasing \(T\) and epochs consistently improved performance, real-world scenarios might necessitate limiting these parameters for time-sensitive applications (eg. T=5 took 50 minutes to train, T=10 took 4 hours). .


Overall, our results confirm that *SAMME* with a carefully tuned Perceptron weak learner can achieve high accuracy and F1-scores on the digits dataset, with the re-weighting effectively alleviating hard misclassifications over multiple rounds.

= Conclusion

In this project, we successfully applied *multi-class AdaBoost (SAMME)* with a *combined Perceptron* as the weak learner to classify handwritten digits. By first selecting the best Perceptron hyperparameters through *k-fold cross-validation* and then training \(T\) boosted rounds, we achieved strong performance on the digits dataset—reaching an accuracy of approximately *97.22%* on the the final test set, with corresponding F1-scores of *0.97*. Before boosting, we had an overall accuracy around 86.11% and F1-score near 0.87, indicating significant improvements over a single Perceptron baseline.



#show: appendix
= Appendix
- Project Repo: https://github.com/RizPur/MALIS-Project/tree/main/
