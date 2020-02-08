drop view if exists tenth_attempt; 

create view tenth_attempt (kicker, seas, wk, pid, kicks)
as
select f.fkicker, g.seas, g.wk, f.pid, count(*) as kicks
from fgxp f
left join play p on f.pid = p.pid
left join game g on p.gid = g.gid
where f.pid  in (
select ff.pid
from fgxp ff
left join play pp on ff.pid = pp.pid
left join game gg on gg.gid = pp.gid
where ff.fkicker = f.fkicker
order by gg.seas, gg.wk ASC
limit 10
)
group by f.fkicker
having MAX(f.pid) and count(*)=10


-- this currently gets the 10th attempt