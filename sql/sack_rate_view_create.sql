drop view if exists sack_rate;
create view sack_rate as
select g.seas as year, pb.psr as player, sum(p.sk)/count(*) as sack_rate
from play p
join pbp pb on p.pid=pb.pid
join game g on p.gid=g.gid
where pb.psr is not null and pb.psr!=''
group by pb.psr,g.seas;