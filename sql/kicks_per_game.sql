-- kicks per game 
select avg(x.cnt), std(x.cnt) from (select count(*) as cnt, gid from play join fgxp on fgxp.pid=play.pid group by gid) x