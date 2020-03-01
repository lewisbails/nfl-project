-- games from kickers in kickers table that werent in the fifty table 
select k.player,k.seas from kicker k where k.player not in (select fkicker from fifty);