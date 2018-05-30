# The objective of this assignment is built around document clustering. 
# The K-Means algorithm may find a suboptimal solution when the centers are chosen badly. 
# The problem chosen for analysis is the adoption of genetic approach to a K-Means clustering algorithm 
# such that we obtain better centroids and Davies-Bouldin index. 
# The implementation is a comparative study between a genetic spherical K-means algorithm 
# (an internal R library) and a self-implemented genetic approach K -means. 
# To understand the performance of the later, we plot the clusters created by both the approaches 
# as a word cloud to analyze the captured core context of the clusters by the approaches.

# Libraries to perform this exercise

library(genalg)
library(ggplot2)
library(GA)     
library(clusterSim)
library(tidyverse)     
library(stringr)       
library(tidytext)
library(tm)
library(SnowballC)
library(wordcloud)
library(ggplot2)
library(ggdendro)
library(dplyr)
library(cluster)
library(HSAUR)
library(fpc)
library(skmeans)
library(plyr)
library(philentropy)
library(gplots)
library(stats)

corpus <- VCorpus(DirSource("corpus", recursive = TRUE, encoding = "UTF-8"), readerControl = list(language = "eng"))
corpus_bk <- corpus
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))

corpus <- tm_map(corpus, toSpace, "/")
corpus <- tm_map(corpus, toSpace, "/.")
corpus <- tm_map(corpus, toSpace, "@")
corpus <- tm_map(corpus, toSpace, "\\|")

# Convert the text to lower case
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, removePunctuation)

# Remove numbers
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, c(letters)) 
corpus <- tm_map(corpus, stemDocument)
corpus.dtm <- DocumentTermMatrix(corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
corpus.dtm<-removeSparseTerms(corpus.dtm, 0.999)
corpus.dtm.mat <- corpus.dtm %>% as.matrix()

# remove any zero rows
corpus.dtm.mat <- corpus.dtm.mat[rowSums(corpus.dtm.mat^2) !=0,]
percent = 10
sample_size = nrow(corpus.dtm.mat) * percent/100
corpus.dtm.mat.sample <- corpus.dtm.mat[sample(1:nrow(corpus.dtm.mat), sample_size, replace=FALSE),]
k=2

#call the skmeans function, it returns a vector of cluster assignments
corpus.dtm.mat.sample.skm  <- skmeans(corpus.dtm.mat.sample,k, method='genetic')

# we convert the vector to a data frame and give it meaninful columns names
corpus.dtm.mat.sample.skm <- as.data.frame(corpus.dtm.mat.sample.skm$cluster)
colnames(corpus.dtm.mat.sample.skm) = c("cluster")

# first, create a tdm weighted by term frequency (tf). 
corpus.tdm <- TermDocumentMatrix(corpus, control = list(weighting = function(x) weightTf(x)))

# remove the sparse terms
corpus.tdm<-removeSparseTerms(corpus.tdm, 0.999)

# select only the documents from the  random sample taken earlier
corpus.tdm.sample <- corpus.tdm[, rownames(corpus.dtm.mat.sample)]

# convert to r matrix
corpus.tdm.sample.mat <- corpus.tdm.sample %>% as.matrix()

# number of clusters
m<- length(unique(corpus.dtm.mat.sample.skm$cluster))
set.seed(2474)
par(mfrow=c(2,3))

# for each cluster plot an explanatory word cloud
for (i in 1:m) 
{
   # the documents in  cluster i
   cluster_doc_ids <-which(corpus.dtm.mat.sample.skm$cluster==i)
  
   # the subset of the matrix with these documents
   corpus.tdm.sample.mat.cluster<- corpus.tdm.sample.mat[, cluster_doc_ids]
  
   # sort the terms by frequency for the documents in this cluster
   v <- sort(rowSums(corpus.tdm.sample.mat.cluster),decreasing=TRUE)
   d <- data.frame(word = names(v),freq=v)
   
   # call word cloud function
   wordcloud(words = d$word, freq = d$freq, scale=c(5,.2), min.freq = 3, max.words=60, 
             random.order=FALSE, rot.per=0.35, colors = brewer.pal(8, "Dark2"))
  title(paste("Cluster", i))
}

# Davies-Bouldin index
DBI <- function(x) 
{
  if(sum(x)==nrow(corpus.dtm.mat.sample) | sum(x)==0)
  {
    score <- 0
  } 
  else 
  {
    cl1 <- pam(corpus.dtm.mat.sample,2)
    d<-dist(corpus.dtm.mat.sample)
    dbi <- index.DB(corpus.dtm.mat.sample, cl1$clustering,d, centrotypes = "medoids")
    score <- dbi$DB
  }
  return(-1*score)
}

iter = 5
GAmodel <- rbga.bin(size = nrow(corpus.dtm.mat.sample), popSize = 10, iters = iter, 
                    mutationChance = 0.05, elitism = TRUE, evalFunc = DBI)
bestSolution<-GAmodel$population[which.min(GAmodel$evaluations),]
bestSolution <- bestSolution+1
bestSolution

for (i in 1:m) 
{
  # the documents in  cluster i
  cluster_doc_ids <-which(bestSolution==i)
  
  # the subset of the matrix with these documents
  corpus.tdm.sample.mat.cluster<- corpus.tdm.sample.mat[, cluster_doc_ids]
  
  # sort the terms by frequency for the documents in this cluster
  v <- sort(rowSums(corpus.tdm.sample.mat.cluster),decreasing=TRUE)
  d <- data.frame(word = names(v),freq=v)
  
  # call word cloud function
  wordcloud(words = d$word, freq = d$freq, scale=c(5,.2), min.freq = 3, max.words=60, 
            random.order=FALSE, rot.per=0.35, colors = brewer.pal(8, "Dark2"))
  title(paste("Cluster", i))
}