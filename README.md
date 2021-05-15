# Sentiment Analyzer
No NLTK or Tensorflow or Pytorch, only math functions. 

sentiment.py contains two separate implementations for a sentiment analyzer class, one uses a Naive-Bayes model with a bag-of-words baseline while another also uses a similar base structure with a few extra optimizations in the model with respect to weights inside the model itself. More explanation on these optimizations coming soon. 

TODO: 
- Better CLI to output labels for SentimentAnalyzer
- Integration with web services to create applet. 
- Smarter file scanner to accept more formats for reviews. 
