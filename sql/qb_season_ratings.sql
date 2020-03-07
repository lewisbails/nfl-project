select *
from
    (select o.player as player, p.pname as name, p.fname as first_name, p.lname as last_name, g.seas as year, sum(o.pa) as attempts, sum(o.pc) as completions, sum(o.py) as yards, sum(o.ints) as interceptions, sum(o.tdp) as TD, round((((avg(o.pc)/avg(o.pa))-0.3)*5 + ((avg(o.py)/avg(o.pa)) -3)*0.25 + avg(o.tdp)/avg(o.pa)*20 + 2.375 - (avg(o.ints)/avg(o.pa)*25))/6*100,1) as avg_rating,
        p.height as height, p.weight as weight, p.hand as hand, p.arm as arm, p.dpos as draft, p.start as start, p.forty as forty, p.bench as bench, p.broad as broad, p.shuttle as shuttle, p.cone as cone, p.dob as dob,
        p.dv as college_div
    from offense o
        join game g on g.gid=o.gid
        join player p on p.player=o.player
    where o.pa>0
    group by o.player,g.seas) l
    join sack_rate r
    on l.player=r.player and l.year=r.year
    join condition_rate cr
    on cr.player=r.player and cr.year=r.year
where l.attempts>50
-- and l.hand>0 and l.height>0 and l.arm>0
order by 5 desc;