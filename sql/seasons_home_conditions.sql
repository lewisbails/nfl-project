-- temps, wind, surface for home and away for kickers, showing their team  

select k.player, k.seas, k.team, 
sum(k.pat+k.fgs+k.fgm+k.fgl) as total_attempts, 
round(avg(case when g.H=k.team then g.temp else null end),1) as av_h_temp, 
round(avg(case when g.H!=k.team then g.temp else null end),1) as av_a_temp, 
round(avg(case when g.H=k.team then g.wspd else null end),1) as av_h_wind, 
round(avg(case when g.H!=k.team then g.wspd else null end),1) as av_h_wind,
sum(case when g.H=k.team then 1 else 0 end) as home,
sum(case when g.surf!='grass' and g.H=k.team then 1 else 0 end) as home_turf,
sum(case when g.H!=k.team then 1 else 0 end) as away,
sum(case when g.surf!='grass' and g.H!=k.team then 1 else 0 end) as away_turf,
sum(case when g.surf='grass' and g.H!=k.team then 1 else 0 end) as away_grass,
round(avg(g.H=k.team),2) as home_p
from kicker k 
left join game g on g.gid=k.gid
group by k.player, k.seas 
order by k.seas desc 
limit 50; 