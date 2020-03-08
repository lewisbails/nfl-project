select *
from
(select o.player as player, g.seas as year, 
sum(o.pa) as attempts, 
sum(o.pc) as completions, 
sum(o.py) as yards, 
sum(o.ints) as interceptions, 
sum(o.tdp) as TD, 
round((((avg(o.pc)/avg(o.pa))-0.3)*5 + ((avg(o.py)/avg(o.pa)) -3)*0.25 + avg(o.tdp)/avg(o.pa)*20 + 2.375 - (avg(o.ints)/avg(o.pa)*25))/6*100,1) as avg_rating
from offense o
join game g on g.gid=o.gid
where o.pa>0
group by o.player,g.seas) l
join sack_rate sr
on l.player=sr.player and l.year=sr.year
join condition_rate cr
on cr.player=sr.player and cr.year=sr.year
join player p on p.player=l.player
-- and l.hand>0 and l.height>0 and l.arm>0
order by 5 desc;