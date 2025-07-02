create database police;
use police;

drop table if exists crime_data_history;
drop table if exists crime_data;
drop table if exists shoes_data;

create table crime_data
(
crime_number varchar(100) not null primary key, 
image_number varchar(100) null,
crime_name varchar(300) null,
find_time varchar(300) null,
request_office varchar(300) null,
find_method varchar(300) null,
state integer not null default 0,
ranking integer not null default 0,
matching_shoes varchar(300) null,
top varchar(300) null,
mid varchar(300) null,
bottom varchar(300) null,
outline varchar(3000) null,
edit_image longtext null
);

create table crime_data_history
(id integer auto_increment primary key,
edit_time text null,
register_time datetime not null,
ranking integer not null default 0,
crime_number varchar(100) not null,
matching_shoes varchar(300) null,
top varchar(300) null,
mid varchar(300) null,
bottom varchar(300) null,
outline varchar(3000) null,
edit_image longtext null,
FOREIGN KEY (crime_number) REFERENCES crime_data(crime_number)
);

create table shoes_data
(id integer auto_increment primary key,
find_location varchar(300) null,
find_year int null,
manufacturer varchar(300) null,
model_number varchar(300) null,
emblem varchar(300) null,
top varchar(300) null,
mid varchar(300) null,
bottom varchar(300) null,
outline varchar(3000) null
);


commit;