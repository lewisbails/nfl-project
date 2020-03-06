use nfl;
drop view if exists condition_rates;
create view condition_rate as
select g.seas as year, o.player as player,
avg(case when g.stad like "%Mile High%" then 1 else 0 end) as alt_rate,
avg(case when g.cond like "%Snow%" then 1 when g.cond like "%Rain%" and not "Chance Rain" then 1 else 0 end) as precip_rate,
avg(case g.surf when 'Grass' then 0 else 1 end) as turf_rate,
avg(case when (g.wspd = '') or (g.wspd is null) then 0 else g.wspd end) as wind_rate,
avg(case when g.v=o.team then 1 else 0 end) as away_rate,
avg(case when (g.temp is not null) and (g.temp!='') then g.temp when (g.temp is null or g.temp='') and (g.cond like '%Dome%' or g.cond like '%Closed%') then 70 else null end) as temp
from game g
join offense o on g.gid=o.gid
where o.pa>0
group by o.player,g.seas;