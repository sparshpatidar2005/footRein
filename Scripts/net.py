import numpy as np
from Scripts.PlayersProperties import defaultProperties
from Scripts.regression import Regression
import os
import sys
import time
from copy import deepcopy
from Scripts.q_learning import QLearningAgent

# Keep one Q-table during a running application, even when the UI creates
# a new Environment after a goal/restart.
_SHARED_RL_AGENT = None

class network:

    

    def __init__(self):
        self.count = 0
        self.updateScore = False
        self.TeamAScore = 0
        self.TeamBScore = 0
        self.game_status = ''
                # RL can be disabled to run the original random-passing baseline.
        self.rl_enabled = os.getenv("FOOTREIN_RL", "1") == "1"
        self.evaluation_epsilon = float(os.getenv("FOOTREIN_EPSILON", "0.15"))

        # Reuse the Q-table across Environment instances in this process.
        global _SHARED_RL_AGENT
        if _SHARED_RL_AGENT is None:
            _SHARED_RL_AGENT = QLearningAgent()
        self.rl_agent = _SHARED_RL_AGENT
        if hasattr(self.rl_agent, "epsilon"):
            self.rl_agent.epsilon = self.evaluation_epsilon

        self.pending_rl_transition = None
        self._pass_in_progress = False
        self._interception_recorded_for_pass = False
        self.match_stats = {
            "passes_attempted": 0,
            "passes_completed": 0,
            "interceptions": 0,
            "goals_A": 0,
            "goals_B": 0,
            "reward_sum": 0.0,
        }

        self.playersListA = ['PlayerAKeeper', 'PlayerA1', 'PlayerA2', 'PlayerA3', 'PlayerA4', 'PlayerA5',
                             'PlayerA6', 'PlayerA7', 'PlayerA8', 'PlayerA9', 'PlayerA10']
        self.playersListB = ['PlayerBKeeper', 'PlayerB1', 'PlayerB2', 'PlayerB3', 'PlayerB4', 'PlayerB5',
                             'PlayerB6', 'PlayerB7', 'PlayerB8', 'PlayerB9', 'PlayerB10']

        DefaultCoordinates = {
            'ball': (600, 350), 'PlayerAKeeper': (40, 330), 'PlayerA1': (140, 100),
            'PlayerA2': (140, 250), 'PlayerA3': (140, 400), 'PlayerA4': (140, 550),
            'PlayerA5': (340, 150), 'PlayerA6': (340, 300), 'PlayerA7': (340, 450),
            'PlayerA8': (500, 350), 'PlayerA9': (550, 300), 'PlayerA10': (550, 400),
            'PlayerB1': (1030, 100), 'PlayerB2': (1030, 250), 'PlayerB3': (1030, 400),
            'PlayerB4': (1030, 550), 'PlayerB5': (830, 150), 'PlayerB6': (830, 300),
            'PlayerB7': (830, 450), 'PlayerB8': (750, 350), 'PlayerB9': (650, 300),
            'PlayerB10': (650, 400), 'PlayerBKeeper': (1130, 330)
        }

        self.superDict = {}
        # initialize ball
        self.superDict['ball'] = defaultProperties()
        self.superDict['ball'].setCoordinates(*DefaultCoordinates['ball'])

        # initialize playersA
        for k in self.playersListA:
            self.superDict[str(k)] = defaultProperties()
            self.superDict[str(k)].setCoordinates(*DefaultCoordinates[str(k)])

        # initialize playersB
        for k in self.playersListB:
            self.superDict[str(k)] = deepcopy(defaultProperties())
            self.superDict[str(k)].setCoordinates(*DefaultCoordinates[str(k)])

        x, y = 0, 0
        return

    def sortdata(self, dictData):
        vals = list(dictData.values())
        keys = list(dictData.keys())
        vals.sort()
        sortedKeys = []
        for i in vals:
            for j in keys:
                if dictData[j] == i:
                    sortedKeys.append(j)

        return sortedKeys

    def StraightLineDistance(self, x1, x2, y1, y2):
        # returns the straight line distance between coordinate (x1, y1) and coordinate (x2, y2)
        x = pow(np.abs(x1 - x2), 2)
        y = pow(np.abs(y1 - y2), 2)

        return np.sqrt(x+y)

    
    def _resolve_pending_transition(self, reward, terminal=False):
        """Update the current Q-learning decision from an observed outcome."""
        pending = self.pending_rl_transition
        if pending is None:
            return

        team = pending["team"]
        ball_x, ball_y = self.GetplayersLocation("ball")
        goal_x = 1160 if team == "A" else 20

        opponent_names = (
            self.playersListB[1:] if team == "A"
            else self.playersListA[1:]
        )
        opponent_distances = []
        for name in opponent_names:
            ox, oy = self.GetplayersLocation(name)
            opponent_distances.append(
                self.StraightLineDistance(ball_x, ox, ball_y, oy)
            )
        pressure = min(opponent_distances, default=200.0)

        next_state = self.rl_agent.encode_state(
            ball_x, ball_y, pressure, abs(goal_x - ball_x)
        )

        self.rl_agent.learn(
            pending["state"],
            pending["action"],
            float(reward),
            next_state,
            pending["actions"],
            terminal=terminal
        )
        self.match_stats["reward_sum"] += float(reward)
        self.pending_rl_transition = None

    def _finish_goal_transition(self, scoring_team):
        """Give the pending action a reward based on the actual score event."""
        pending = self.pending_rl_transition
        if pending is not None:
            reward = 10.0 if pending["team"] == scoring_team else -10.0
            self._resolve_pending_transition(reward, terminal=True)
        self._pass_in_progress = False

    def complete_pass_if_no_interception(self):
        """Record a completed pass at the end of its movement segment.

        This simulator has no separate ball-receiver collision event, so a pass
        is considered completed if no interception was detected during the
        segment and the segment did not end in a goal/keeper restart.
        """
        if not self._pass_in_progress:
            return

        if not self._interception_recorded_for_pass:
            self.match_stats["passes_completed"] += 1
            pending = self.pending_rl_transition
            if pending is not None:
                team = pending["team"]
                goal_x = 1160 if team == "A" else 20
                ball_x, _ = self.GetplayersLocation("ball")
                distance_now = abs(goal_x - ball_x)
                progress = (pending["goal_distance"] - distance_now) / 100.0
                progress = max(-1.0, min(1.0, progress))
                # Outcome reward is based on completed possession plus
                # measured progress; no hypothetical outcome is assumed.
                self._resolve_pending_transition(1.0 + progress)

        self._pass_in_progress = False

    def ComputeGoal(self):
        # get KeeperA's location
        x1, y1 = self.GetplayersLocation(defaultPlayer='PlayerAKeeper')
        x2, y2 = self.GetplayersLocation(defaultPlayer='PlayerBKeeper')
        ballx, bally = self.superDict['ball'].GetlastLocation()
        GoalStatus = 'Nil'
        if ballx >= 20 and ballx <= 40:
            if (bally > y1+20 or bally < y1-20) and (bally > 250 and bally < 450):
                self.game_status = 'Goal!!!!'
                # increment team score
                self.TeamBScore += 1
                self.match_stats['goals_B'] += 1
                self._finish_goal_transition('B')
                out = True
                GoalStatus = 'Nil'

            else:
                GoalStatus = 'PlayerAKeeper'
                out = False

        elif ballx >= 1130 and ballx <= 1180:
            if (bally > y2+20 or bally < y2-20) and (bally > 250 and bally < 450):
                self.game_status = 'Goal!!!!'
                # increment team score
                self.TeamAScore += 1
                self.match_stats['goals_A'] += 1
                self._finish_goal_transition('A')
                out = True
            else:
                GoalStatus = 'PlayerBKeeper'
                out = False
        else:
            GoalStatus = 'Nil'
            out = False
        return out, GoalStatus

    def GetPostBoundary(self, keeper='PlayerBKeeper'):
        if keeper == 'PlayerBKeeper':
            ytarget = np.random.choice([i for i in range(250, 450)])
            xtarget = 1160

        else:
            ytarget = np.random.choice([i for i in range(250, 450)])
            xtarget = 20
        return xtarget, ytarget

    def AttemptScore(self, player, keeper, XDict, YDict):
        xtarget, ytarget = self.GetPostBoundary(keeper=keeper)
        xsource, ysource = self.GetplayersLocation(defaultPlayer=player)
        xmoves, ymoves = self.linearRegression(
            xsource, ysource, xtarget, ytarget)

        # set balls trajectory
        self.setplayersLocation(coordinates=(
            xmoves, ymoves), defaultPlayer='ball')

        # decide if the keeper should get the ball or loose it
        keeperStatus = np.random.choice([True, False], p=[0.7, 0.3])
        if keeperStatus == True:
            # set Keeper's movement to follow the ball

            if keeper in self.playersListA:
                self.superDict['PlayerAKeeper'].listY = [ymoves[-1]-10, ymoves[-1]-5,
                                                         ymoves[-1]+10, ymoves[-1]+5, ymoves[-1]]*10
                self.superDict['PlayerAKeeper'].listX = [xmoves[-1]]*50
                self.game_status = "Awesome!!! Goalkeeper caught the ball."

            else:
                self.superDict['PlayerBKeeper'].listY = [ymoves[-1]-10, ymoves[-1]-5,
                                                         ymoves[-1]+10, ymoves[-1]+5, ymoves[-1]]*10
                self.superDict['PlayerBKeeper'].listX = [xmoves[-1]]*50
                self.game_status = "Awesome!!! Goalkeeper caught the ball."

        else:
            currentPositionX, currentPositionY = self.GetplayersLocation(
                defaultPlayer=keeper)
            xbound, ybound = self.GetplayersBoundary(keeper)

            xmoves, ymoves = self.linearRegression(currentPositionX, currentPositionY,
                                                   xbound, ybound)
            self.setplayersLocation((xmoves, ymoves), defaultPlayer=keeper)

        if keeper in list(XDict.keys()):
            XDict.pop(keeper)
        else:
            YDict.pop(keeper)

        # Re-position other players
        for playee in XDict.keys():
            currentPositionX, currentPositionY = XDict[playee]
            xbound, ybound = self.GetplayersBoundary(playee)

            xmoves, ymoves = self.linearRegression(currentPositionX, currentPositionY,
                                                   xbound, ybound)
            self.setplayersLocation((xmoves, ymoves), defaultPlayer=playee)

        for playee in YDict.keys():
            currentPositionX, currentPositionY = YDict[playee]
            xbound, ybound = self.GetplayersBoundary(playee)
            xmoves, ymoves = self.linearRegression(currentPositionX, currentPositionY,
                                                   xbound, ybound)
            self.setplayersLocation((xmoves, ymoves), defaultPlayer=playee)

        return

    def Attack(self, OpposingPlayers):
        """
        Return the next player in the opposing team close enough to get the ball from the
        current team.
        """
        playee = 'A'
        Status = False
        # Get the current ball location
        count = self.count
        currentX, currentY = self.superDict['ball'].listX[count], self.superDict['ball'].listY[count]

        EstimatedDistance = {}
        for player in OpposingPlayers.keys():
            # find Euclidean Distance from ball
            playeeCoordinate = OpposingPlayers[player]
            EstimatedDistance[player] = self.StraightLineDistance(playeeCoordinate[0],
                                                                  currentX, playeeCoordinate[1], currentY)

        sortedKey = self.sortdata(EstimatedDistance)[0]
        if EstimatedDistance[sortedKey] <= 50:
            Status = True
            playee = sortedKey

            # Record one interception at most for the current pass.
            if self._pass_in_progress and not self._interception_recorded_for_pass:
                self._interception_recorded_for_pass = True
                self.match_stats["interceptions"] += 1
                self._resolve_pending_transition(-1.0)
        else:
            Status = False
            playee = 'A'
        return playee, Status

    def TagPlayers(self, XDict, YDict):
        """
        YDict in this case represents the source players
        XDict in this case represents the target players
        That is, YDict are tagging each XDict Player (excluding goal keepers)
        """
        SourcePlayers = list(YDict.keys())
        TargetPlayers = list(XDict.keys())

        # remove goal keepers
        # goalkeepers are always first on the list
        TargetPlayers.remove(TargetPlayers[0])
        # goalkeepers are always first on the list
        SourcePlayers.remove(SourcePlayers[0])

        # Tag players to players based on nearest neighbors
        DictTags = {}

        for k in SourcePlayers:
            EstimatedDistance = {}
            for j in TargetPlayers:
                # get source location
                xsource, ysource = self.GetplayersLocation(k)
                # get target location
                xtarget, ytarget = self.GetplayersLocation(j)

                EstimatedDistance[j] = self.StraightLineDistance(
                    xsource, xtarget, ysource, ytarget)
            FirstKey = self.sortdata(EstimatedDistance)[0]
            DictTags[k] = FirstKey
            # remove the chosen key from this data.
            TargetPlayers.remove(FirstKey)

        # reset movement of source players to target players
        for keys in DictTags.keys():
            value = DictTags[keys]
            # get source location
            xsource, ysource = self.GetplayersLocation(keys)
            xtarget, ytarget = self.GetplayersLocation(value)

            xmoves, ymoves = self.linearRegression(
                xsource, ysource, xtarget, ytarget)

            # reposition the ball in front of the player

            self.setplayersLocation(coordinates=(
                xmoves, ymoves), defaultPlayer=keys)

        return

    
    def LikePairsNearestNeighbor(
        self,
        playername,
        sourcePlayerCoordinate,
        SourcedictDataX,
        SourcedictDataY,
        milestone='short',
        conditionedValue=30
    ):
        """
        RL-based pass selection.

        State:
            Ball location, vertical lane, opponent pressure,
            and distance to the opponent's goal.

        Actions:
            nearest, forward, open

        Reward:
            Outcome-based reward from pass completion, interception, or goal.
        """

        # Baseline mode preserves the original strategy: randomly choose a
        # teammate from the distance-sorted list, excluding the current player.
        if not self.rl_enabled:
            distances = {
                name: self.StraightLineDistance(
                    sourcePlayerCoordinate[0], coordinate[0],
                    sourcePlayerCoordinate[1], coordinate[1]
                )
                for name, coordinate in SourcedictDataX.items()
            }
            sorted_players = self.sortdata(distances)
            if playername in sorted_players:
                sorted_players.remove(playername)
            if not sorted_players:
                return playername
            self.game_status = "Baseline random pass"
            return np.random.choice(sorted_players)

        # Possible receivers on the same team.
        candidates = [
            name for name in SourcedictDataX
            if name != playername
        ]

        if not candidates:
            return playername

        # Current ball position.
        ball_x, ball_y = self.GetplayersLocation('ball')

        # Team A attacks right; Team B attacks left.
        team_a = playername.startswith('PlayerA')
        goal_x = 1160 if team_a else 20

        current_goal_distance = abs(goal_x - ball_x)

        # Positions of the opposing team.
        opponents = list(SourcedictDataY.values())

        # Distance from the ball to the nearest opponent.
        pressure = min(
            (
                self.StraightLineDistance(
                    ball_x, ox, ball_y, oy
                )
                for ox, oy in opponents
            ),
            default=200
        )

        # Encode the current situation as an RL state.
        state = self.rl_agent.encode_state(
            ball_x,
            ball_y,
            pressure,
            current_goal_distance
        )

        # Calculate distances from the current player
        # to every potential receiver.
        distances = {}

        for name in candidates:
            x, y = SourcedictDataX[name]

            distances[name] = self.StraightLineDistance(
                sourcePlayerCoordinate[0],
                x,
                sourcePlayerCoordinate[1],
                y
            )

        def target_metrics(name):
            x, y = SourcedictDataX[name]

            # Distance between receiver and opponent's goal.
            goal_distance = abs(goal_x - x)

            # Distance from receiver to nearest opponent.
            opponent_distance = min(
                (
                    self.StraightLineDistance(
                        x, ox, y, oy
                    )
                    for ox, oy in opponents
                ),
                default=200
            )

            # Positive means the receiver is closer to goal.
            forward_progress = (
                current_goal_distance - goal_distance
            )

            # Normalize openness to a range of 0 to 1.
            openness = min(
                opponent_distance, 150.0
            ) / 150.0

            return forward_progress, openness

        # Available RL actions.
        actions = ('nearest', 'forward', 'open')

        
        # The agent chooses which passing tactic to use.
        selected_action = self.rl_agent.choose_action(
            state, actions
        )

        if selected_action == 'nearest':

            # Pass to the closest teammate.
            nextplayer = min(
                candidates,
                key=lambda name: distances[name]
            )

        elif selected_action == 'forward':

            # Pass to the teammate making the most
            # forward progress toward the goal.
            nextplayer = max(
                candidates,
                key=lambda name: target_metrics(name)[0]
            )

        else:

            # Pass to the teammate with the most space
            # from the nearest opponent.
            nextplayer = max(
                candidates,
                key=lambda name: target_metrics(name)[1]
            )

        # Record the decision. Its reward will be assigned
        # after the next possession outcome is observed.
        self.pending_rl_transition = {
            'state': state,
            'action': selected_action,
            'team': 'A' if playername.startswith('PlayerA') else 'B',
            'goal_distance': current_goal_distance,
            'actions': actions
        }

        self.game_status = (
            f'RL pass: {selected_action} -> {nextplayer}'
        )

        return nextplayer


    def repositionBall(self, xmove, ymove):
        i = 0
        x = [i+1 for i in xmove]
        y = [i+1 for j in ymove]
        return x, y

    def AttemptGoal(self, playername, sourcePlayerCoordinate, SourcedictDataX, SourcedictDataY):
        # Attempt Shot
        # get opposing keeper, target Team.
        if playername in self.playersListA:
            goalKeeper = 'PlayerBKeeper'
        else:
            goalKeeper = 'PlayerAKeeper'

        self.game_status = playername + \
            ' is Attempting to make Shot, goalkeeper is ' + goalKeeper
        self.AttemptScore(playername, goalKeeper,
                          SourcedictDataX, SourcedictDataY)
        return

    def updateposition(self, playername, sourcePlayerCoordinate, SourcedictDataX, SourcedictDataY):

        self.match_stats["passes_attempted"] += 1
        self._pass_in_progress = True
        self._interception_recorded_for_pass = False

        # get the next player
        outplayer = self.LikePairsNearestNeighbor(
            playername, sourcePlayerCoordinate, SourcedictDataX, SourcedictDataY)

        # update straight line movement of the ball from source to nextplayer

        xbound, ybound = SourcedictDataX[outplayer]
        xsource, ysource = self.superDict['ball'].GetlastLocation()

        xmoves, ymoves = self.linearRegression(
            xsource, ysource, xbound, ybound)

        # reposition the ball in front of the player

        self.setplayersLocation(coordinates=(
            xmoves, ymoves), defaultPlayer='ball')
        self.setplayersLocation(coordinates=(
            [xbound]*50, [ybound]*50), defaultPlayer=outplayer)

        # remove playername, and update new player
        SourcedDictDataX_copy = SourcedictDataX.copy()
        SourcedDictDataX_copy.pop(outplayer)

        # update the positions of all players including playername
        """
        1. get their current position from the sourcedictDataX and SourcedDictDataY
        2. estimate a straight line move to their new position (within the specified boundary)
        """
        for playee in SourcedDictDataX_copy.keys():
            currentPositionX, currentPositionY = SourcedDictDataX_copy[playee]
            xbound, ybound = self.GetplayersBoundary(playee)

            xmoves, ymoves = self.linearRegression(currentPositionX, currentPositionY,
                                                   xbound, ybound)
            self.setplayersLocation((xmoves, ymoves), defaultPlayer=playee)

        # reset oppossing keepers position
        currentPositionX, currentPositionY = SourcedictDataY[list(
            SourcedictDataY.keys())[0]]
        xbound, ybound = self.GetplayersBoundary(
            list(SourcedictDataY.keys())[0])
        xmoves, ymoves = self.linearRegression(currentPositionX, currentPositionY,
                                               xbound, ybound)
        self.setplayersLocation(
            (xmoves, ymoves), defaultPlayer=list(SourcedictDataY.keys())[0])

        # for the opposing players, reset their positions to attack.
        self.TagPlayers(SourcedictDataX, SourcedictDataY)

        return outplayer, playername

    def linearRegression(self, x1, y1, x2, y2):
        reg = Regression()
        xmove, ymove = reg.network(x1, y1, x2, y2)

        return xmove, ymove

    def setplayersLocation(self, coordinates, defaultPlayer='ball'):
        """
        x and y are lists of 50 different positions for the specific 'player'
        updates a new location for the player passed in as defaultPlayer
        """
        x = coordinates[0]
        y = coordinates[1]

        if defaultPlayer in self.superDict.keys():
            self.superDict[defaultPlayer].listX = x
            self.superDict[defaultPlayer].listY = y
        else:
            pass

        return

    def GetplayersLocation(self, defaultPlayer='ball'):
        """
        updates a new location for the player passed in as defaultPlayer
        """
        x = self.superDict[defaultPlayer].positionx
        y = self.superDict[defaultPlayer].positiony

        return x, y

    def GetplayersBoundary(self, defaultPlayer='ball'):
        """
        Get the boundary's of keepers, ball, defenders, strikers and mid-fielders
        """
        # Determine factor, range of factor is 0 - 1
        factor = 1
        factor2 = 1
        stat = False
        # get ball's coordinate
        xball, yball = self.GetplayersLocation('ball')
        if yball < 330:
            stat = False
            # within player's A post
            factor = 0.2
            factor2 = 1

        elif yball > 330 and yball < 660:
            stat = True
            factor = 1
            factor2 = 1

        else:
            stat = False
            factor = 0.7
            factor2 = 1/factor

        if defaultPlayer in ['PlayerA1', 'PlayerA2', 'PlayerB9']:
            x1, x2, y1, y2 = 150*factor, 600*factor*factor2, 330, 660

        elif defaultPlayer in ['PlayerA3', 'PlayerA4', 'PlayerB10']:
            x1, x2, y1, y2 = 150*factor, 600*factor*factor2, 20, 330

        elif defaultPlayer == 'PlayerB8':
            x1, x2, y1, y2 = 150*factor, 600*factor*factor2, 275, 345

        elif defaultPlayer in ['PlayerA5', 'PlayerB5']:
            x1, x2, y1, y2 = 400*factor, 900*factor*factor2, 330, 660

        elif defaultPlayer in ['PlayerA6', 'PlayerB6']:
            x1, x2, y1, y2 = 400*factor, 900*factor*factor2, 275, 345

        elif defaultPlayer in ['PlayerA7', 'PlayerB7']:
            x1, x2, y1, y2 = 400*factor, 900*factor*factor2, 20, 330

        elif defaultPlayer == 'PlayerA8':
            x1, x2, y1, y2 = 600*factor, 1000*factor*factor2, 275, 345

        elif defaultPlayer == 'PlayerA9':
            x1, x2, y1, y2 = 600*factor, 1000*factor*factor2, 330, 660

        elif defaultPlayer == 'PlayerA10':
            x1, x2, y1, y2 = 600*factor, 1000*factor*factor2, 20, 330

        elif defaultPlayer in ['PlayerB1', 'PlayerB2']:
            x1, x2, y1, y2 = 600*factor, 1000*factor*factor2, 330, 660

        elif defaultPlayer in ['PlayerB3', 'PlayerB4']:
            x1, x2, y1, y2 = 600*factor, 1000*factor*factor2, 20, 330

        elif defaultPlayer == 'PlayerAKeeper':
            x1, x2, y1, y2 = 20, 220, 200, 500
        elif defaultPlayer == 'PlayerBKeeper':
            x1, x2, y1, y2 = 1000, 1100, 200, 500
        else:
            x1, x2, y1, y2 = 20, 1160, 20, 660

        # Get player's location to minimize movement around the field
        xloc, yloc = self.GetplayersLocation(defaultPlayer)
        xchoice = np.random.choice([i for i in range(int(x1), int(x2))])
        ychoice = np.random.choice([i for i in range(int(y1), int(y2))])

        # ensure xsource isn't too far from Xnew
        # if stat == True:
        if xloc > xchoice:
            if np.abs(xloc-xchoice) > 100:
                xchoice = xloc - 100

        else:
            if np.abs(xloc-xchoice) > 100:
                xchoice = xloc + 100
        # else:
        #     pass

        # return a random coordinate between x1 and x2, likewise, y1 and y2.

        return xchoice, ychoice
