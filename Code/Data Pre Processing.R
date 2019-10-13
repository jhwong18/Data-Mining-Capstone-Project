# R-CODE FOR DATA PROCESSING
# Data Processing:

setwd("C:/Users/Lim/Desktop/Lecture Notes & Tutorials/AY 2018-19/MH4510 Statistical Learning and Data Mining/Project")
song = read.csv("songs.csv", header = TRUE, sep=",")
song_extra = read.csv("song_extra_info.csv", header = TRUE, sep="," , na.strings = c("", "NA"))
df = read.csv("train.csv", header = TRUE, sep=",")
members = read.csv("members.csv",header = TRUE, sep=",")
song_extra$name <- NULL
song_extra = na.omit(song_extra)
song_extra[] <- lapply(song_extra, as.character)
song_extra$year = substr(song_extra$isrc, 6,7)
song_extra$isrc <- NULL
song_extra_cleaned = song_extra[song_extra$year<=18,]
song_extra_cleaned = song_extra[song_extra$year>=10,]
song_merged = merge(song,song_extra_cleaned)
df_m = merge(df, song_merged)
df_m = merge(df_m, members)
df_m = df_m[df_m$expiration_date>=20180101,]

df_m$registered_via = NULL
df_m$registration_init_time = NULL
df_m$expiration_date = NULL
df_m$lyricist = NULL

df_m = na.fill(df_m, "unknown")
df_m = df_m[df_m$bd >= 6]
df_m = df_m[df_m$bd <= 90]

#split train and test
set.seed(2018)
train <- sample(nrow(df_m), 0.7 * nrow(df_m))
df.train = df_m[train,]
df.test = df_m[-train,]

count1 = aggregate(target ~ artist_name, df.train, mean)
names(count1) = c("artist_name","artist_repeat_percentage")
df.train = merge(df.train, count1)
df.test = merge(df.test, count1)
df.train$artist_name = NULL
df.test$artist_name = NULL

count1 = aggregate(target ~ msno, df.train, mean)
names(count1) = c("msno","msno_repeat_percentage")
df.train = merge(df.train, count1)
df.train$msno = NULL
df.test = merge(df.test, count1)
df.test$msno = NULL

count1 = aggregate(target ~ song_id, df.train, mean)
names(count1) = c("song_id","song_repeat_percentage")
df.train = merge(df.train, count1)
df.train$song_id = NULL
df.test = merge(df.test, count1)
df.test$song_id = NULL

count1 = aggregate(target ~ genre_ids, df.train, mean)
names(count1) = c("genre_ids","genre_ids_repeat_percentage")
df.train = merge(df.train, count1)
df.train$genre_ids = NULL
df.test = merge(df.test, count1)
df.test$genre_ids = NULL

count1 = aggregate(target ~ composer, df.train, mean)
names(count1) = c("composer","composer_repeat_percentage")
df.train = merge(df.train, count1)
df.train$composer = NULL
df.test = merge(df.test, count1)
df.test$composer = NULL

train.label = df.train$target
names(train.label) = "target"
#df.train$target = NULL
test.label = df.test$target
names(test.label) = "target"
df.test$target = NULL

write.csv(df.train, file = "train_data.csv",row.names=FALSE)
write.csv(df.test, file = "test_data.csv", row.names=FALSE)
write.csv(train.label, file = "train_labels.csv", row.names=FALSE)
write.csv(test.label, file = "test_labels.csv", row.names=FALSE)

#split train and test
set.seed(2018)
train <- sample(nrow(df_m), 1 * nrow(df_m))
df.train = df_m[train,]
df.test = df_m[-train,]


count1 = aggregate(target ~ artist_name, df.train, mean)
names(count1) = c("artist_name","artist_repeat_percentage")
df.train = merge(df.train, count1)
df.test = merge(df.test, count1)
df.train$artist_name = NULL
df.test$artist_name = NULL

count1 = aggregate(target ~ msno, df.train, mean)
names(count1) = c("msno","msno_repeat_percentage")
df.train = merge(df.train, count1)
df.train$msno = NULL
df.test = merge(df.test, count1)
df.test$msno = NULL

count1 = aggregate(target ~ song_id, df.train, mean)
names(count1) = c("song_id","song_repeat_percentage")
df.train = merge(df.train, count1)
df.train$song_id = NULL
df.test = merge(df.test, count1)
df.test$song_id = NULL

count1 = aggregate(target ~ genre_ids, df.train, mean)
names(count1) = c("genre_ids","genre_ids_repeat_percentage")
df.train = merge(df.train, count1)
df.train$genre_ids = NULL
df.test = merge(df.test, count1)
df.test$genre_ids = NULL

count1 = aggregate(target ~ composer, df.train, mean)
names(count1) = c("composer","composer_repeat_percentage")
df.train = merge(df.train, count1)
df.train$composer = NULL
df.test = merge(df.test, count1)
df.test$composer = NULL

train.label = df.train$target
names(train.label) = "target"
#df.train$target = NULL
test.label = df.test$target
names(test.label) = "target"
df.test$target = NULL

write.csv(df.train, file = "train_data.csv",row.names=FALSE)
write.csv(df.test, file = "test_data.csv", row.names=FALSE)
write.csv(train.label, file = "train_labels.csv", row.names=FALSE)
write.csv(test.label, file = "test_labels.csv", row.names=FALSE)
write.csv(df.test, file = "test_d.csv", row.names=FALSE)
write.csv(test.label, file = "test_l.csv", row.names=FALSE)

#split train and test
set.seed(2018)
train <- sample(nrow(df_m), 1 * nrow(df_m))
df.train = df_m[train,]
df.test = df_m[-train,]

count1 = aggregate(target ~ artist_name, df.train, mean)
names(count1) = c("artist_name","artist_repeat_percentage")
df.train = merge(df.train, count1)
df.test = merge(df.test, count1)
df.train$artist_name = NULL
df.test$artist_name = NULL

count1 = aggregate(target ~ msno, df.train, mean)
names(count1) = c("msno","msno_repeat_percentage")
df.train = merge(df.train, count1)
df.train$msno = NULL
df.test = merge(df.test, count1)
df.test$msno = NULL

count1 = aggregate(target ~ song_id, df.train, mean)
names(count1) = c("song_id","song_repeat_percentage")
df.train = merge(df.train, count1)
df.train$song_id = NULL
df.test = merge(df.test, count1)
df.test$song_id = NULL

count1 = aggregate(target ~ genre_ids, df.train, mean)
names(count1) = c("genre_ids","genre_ids_repeat_percentage")
df.train = merge(df.train, count1)
df.train$genre_ids = NULL
df.test = merge(df.test, count1)
df.test$genre_ids = NULL

count1 = aggregate(target ~ composer, df.train, mean)
names(count1) = c("composer","composer_repeat_percentage")
df.train = merge(df.train, count1)
df.train$composer = NULL
df.test = merge(df.test, count1)
df.test$composer = NULL

train.label = df.train$target
names(train.label) = "target"
#df.train$target = NULL
test.label = df.test$target
names(test.label) = "target"
df.test$target = NULL

write.csv(df.train, file = "train_data.csv",row.names=FALSE)
write.csv(df.test, file = "test_data.csv", row.names=FALSE)
write.csv(train.label, file = "train_labels.csv", row.names=FALSE)
write.csv(test.label, file = "test_labels.csv", row.names=FALSE)

#split train and test
set.seed(2018)
train <- sample(nrow(df_m), 1 * nrow(df_m))
df.train = df_m[train,]
df.test = df_m[-train,]


count1 = aggregate(target ~ artist_name, df.train, mean)
names(count1) = c("artist_name","artist_repeat_percentage")
df.train = merge(df.train, count1)
df.test = merge(df.test, count1)
df.train$artist_name = NULL
df.test$artist_name = NULL

count1 = aggregate(target ~ msno, df.train, mean)
names(count1) = c("msno","msno_repeat_percentage")
df.train = merge(df.train, count1)
df.train$msno = NULL
df.test = merge(df.test, count1)
df.test$msno = NULL

count1 = aggregate(target ~ song_id, df.train, mean)
names(count1) = c("song_id","song_repeat_percentage")
df.train = merge(df.train, count1)
df.train$song_id = NULL
df.test = merge(df.test, count1)
df.test$song_id = NULL

count1 = aggregate(target ~ genre_ids, df.train, mean)
names(count1) = c("genre_ids","genre_ids_repeat_percentage")
df.train = merge(df.train, count1)
df.train$genre_ids = NULL
df.test = merge(df.test, count1)
df.test$genre_ids = NULL

count1 = aggregate(target ~ composer, df.train, mean)
names(count1) = c("composer","composer_repeat_percentage")
df.train = merge(df.train, count1)
df.train$composer = NULL
df.test = merge(df.test, count1)
df.test$composer = NULL

train.label = df.train$target
names(train.label) = "target"

write.csv(df.train, file = "df.csv",row.names=FALSE)

#load training and test data from csv files
setwd("C:/Users/Lim/Desktop/Lecture Notes & Tutorials/AY 2018-19/MH4510 Statistical Learning and Data Mining/Project")
train_data = read.csv("train_data.csv", header = TRUE, sep=",")
train_data$target = NULL
train_label = read.csv("train_labels.csv", header = TRUE, sep=",")
test_data = read.csv("test_data.csv", header = TRUE, sep=",")
test_data$target = NULL
test_label = read.csv("test_labels.csv", header = TRUE, sep=",")

# encode
train_data$source_system_tab = as.factor(train_data$source_system_tab)
train_data$source_screen_name = as.factor(train_data$source_screen_name)
train_data$source_type = as.factor(train_data$source_type)
train_data$language = as.factor(train_data$language)
train_data$city = as.factor(train_data$city)
train_data$gender = as.factor(train_data$gender)
test_data$source_system_tab = as.factor(test_data$source_system_tab)
test_data$source_screen_name = as.factor(test_data$source_screen_name)
test_data$source_type = as.factor(test_data$source_type)
test_data$language = as.factor(test_data$language)
test_data$city = as.factor(test_data$city)
test_data$gender = as.factor(test_data$gender)

train_data$song_length = as.numeric(train_data$song_length)
train_data$year = as.numeric(train_data$year)
train_data$bd = as.numeric(train_data$bd)
train_data$artist_repeat_percentage = as.numeric(train_data$artist_repeat_percentage)
train_data$msno_repeat_percentage = as.numeric(train_data$msno_repeat_percentage)
train_data$song_repeat_percentage = as.numeric(train_data$song_repeat_percentage)
train_data$genre_ids_repeat_percentage = as.numeric(train_data$genre_ids_repeat_percentage)
train_data$composer_repeat_percentage = as.numeric(train_data$composer_repeat_percentage)
test_data$song_length = as.numeric(test_data$song_length)
test_data$year = as.numeric(test_data$year)
test_data$bd = as.numeric(test_data$bd)
test_data$artist_repeat_percentage = as.numeric(test_data$artist_repeat_percentage)
test_data$msno_repeat_percentage = as.numeric(test_data$msno_repeat_percentage)
test_data$song_repeat_percentage = as.numeric(test_data$song_repeat_percentage)
test_data$genre_ids_repeat_percentage = as.numeric(test_data$genre_ids_repeat_percentage)
test_data$composer_repeat_percentage = as.numeric(test_data$composer_repeat_percentage)
