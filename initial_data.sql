create database police;
use police;

drop table if exists crimeDataHistory;
drop table if exists crimeData;
drop table if exists shoesData;

create table crimeData
(
crimeNumber varchar(100) not null primary key, 
image longtext not null,
imageNumber varchar(100) null,
crimeName varchar(300) null,
findTime varchar(300) null,
requestOffice varchar(300) null,
findMethod varchar(300) null,
state integer not null default 0,
ranking integer not null default 0
matchingShoes varchar(300) null,
);

truncate table crimeData;

create table crimeDataHistory
(id integer auto_increment primary key,
image text not null,
registerTime datetime not null,
ranking integer not null default 0,
crimeNumber varchar(100) not null,
top varchar(300) null,
mid varchar(300) null,
bottom varchar(300) null,
outline varchar(3000) null,
FOREIGN KEY (crimeNumber) REFERENCES crimeData(crimeNumber)
);

truncate table crimeDataHistory;

create table shoesData
(id integer auto_increment primary key,
image text not null,
findLocation varchar(300) null,
manufacturer varchar(300) null,
modelNumber varchar(300) null,
emblem varchar(300) null,
top varchar(300) null,
mid varchar(300) null,
bottom varchar(300) null,
outline varchar(3000) null
);

truncate table shoesData;

commit;