{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ViewPatterns #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module Main where

import AI.NeuralNetwork
import Control.Monad
import Data.Bifunctor
import Data.List
import Data.Maybe
import Graphics.Gloss.Geometry.Line
import Graphics.Gloss.Interface.IO.Game
import System.Console.ANSI
import System.IO
import Text.Printf
import Text.Read (readMaybe)

data State = State
    { points :: [((Float, Float), Int)]
    , net :: Network
    , hidden :: Int
    , err :: Double
    , activationFns :: ActivationFns
    , rate :: Double
    , running :: Bool
    , windowWidth :: Int
    , windowHeight :: Int
    , zoom :: Float
    , pointRadius :: Float }

data PointsPreset = And | Or | Xor | Custom deriving (Show, Enum, Bounded)
data ActivationFnType = Sigmoid | TanH deriving (Show, Enum, Bounded)

main :: IO ()
main = do
    putStrLn "To use the default value, just press enter"
    pointsPreset <- askEnum "Choose points" Custom
    hidden <- ask "Hidden neurons" 2
    net <- network 2 [hidden] 1
    activationFn <- askEnum "Activation function" Sigmoid
    rate <- ask "Learning rate" 1
    fps <- ask "Frames per second" 60
    windowWidth <- ask "Window width" 600
    windowHeight <- ask "Window height" 600
    zoom <- ask "Zoom" $ fromIntegral (min windowWidth windowHeight) / 4
    pointRadius <- ask "Point radius" 10
    running <- askEnum "Initially running" True
    let initState = State
            { points = case pointsPreset of
                And -> zip boolPoints [0, 0, 0, 1]
                Or -> zip boolPoints [0, 1, 1, 1]
                Xor -> zip boolPoints [0, 1, 1, 0]
                Custom -> []
            , activationFns = case activationFn of
                Sigmoid -> (sigmoid, sigmoid')
                TanH -> (tanh, tanh')
            , err = 0
            , .. }
        boolPoints = [(0, 0), (0, 1), (1, 0), (1, 1)]
        window = InWindow "Neural network" (windowWidth, windowHeight) (1300 - windowWidth, 0)
    logStateInit initState
    playIO window black fps initState plot listener $ const update
    where
    ask question def = do
        printf "%v (default is %v): " question def
        hFlush stdout
        fromMaybe def . readMaybe <$> getLine
    askEnum :: forall a. (Show a, Enum a, Bounded a) => String -> a -> IO a
    askEnum question def = do
        zipWithM_ (\n x -> putStrLn $ show n ++ ". " ++ show x) [1..] [(minBound :: a)..]
        flip fmap (ask question $ fromEnum def + 1) $ \answer ->
            if answer >= 1 && answer <= fromEnum (maxBound :: a) + 1
                then toEnum $ answer - 1 else def

update :: State -> IO State
update State {..}
    | running, not $ null points = logState State
        { net = net', err = err', .. }
    | otherwise = return State {..}
    where
    (net', err') = trainOnce' activationFns rate
        (map (bimap (\(x, y) -> map realToFrac [x, y]) (return . fromIntegral)) points) net

plot :: State -> IO Picture
plot State {..} = return $ Pictures
    $  axis [(-fromIntegral windowWidth / 2, 0), (fromIntegral windowWidth / 2, 0)]
    :  axis [(0, -fromIntegral windowHeight / 2), (0, fromIntegral windowHeight / 2)]
    :  map neuronLine (net !! 0)
    ++ map plotPoint points
    where
    axis = Color white . Line
    neuronLine (map realToFrac -> weights, realToFrac -> bias) =
        Color azure $ Line $ flip map [-1, 1] $ \a ->
            fromMaybe (a * zoom, 0) $ intersectLineLine
                (0, -zoom * bias / last weights)
                (-zoom * bias / head weights, 0)
                (0, a * fromIntegral windowHeight / 2)
                (1, a * fromIntegral windowHeight / 2)
    plotPoint ((x, y), t) = Color (if toEnum t then green else red)
        $ Translate (x * zoom) (y * zoom) $ circleSolid pointRadius

listener :: Event -> State -> IO State
listener event state = listener' state event >>= logState

listener' :: State -> Event -> IO State
listener' State {..} (EventKey key Down _ (join bimap (/ zoom) -> (x, y)))
    | MouseButton LeftButton <- key = return $ point 1
    | MouseButton RightButton <- key = return $ point 0
    | SpecialKey KeySpace <- key = return State { running = not running, .. }
    | Char 'l' <- key = return State { rate = rate + 0.1, .. }
    | Char ';' <- key = return State { rate = rate - 0.1, .. }
    | Char 'r' <- key = do
        net' <- network 2 [hidden] 1
        return State { net = net', .. }
    | Char 'c' <- key = return State { points = [], .. }
    | Char c <- key
    , Just n <- readMaybe [c] = do
        let hidden' = if n == 0 then 10 else n
        net' <- network 2 [hidden'] 1
        return State { net = net', hidden = hidden', .. }
    where
    point n = State
        { points = case find inPoint points of
            Just pt -> delete pt points
            Nothing -> ((x, y), n) : points
        , .. }
    inPoint ((x1, y1), _) = sqrt ((y - y1) ^ 2 + (x - x1) ^ 2) < pointRadius / zoom
listener' state _ = return state

controls, variables :: [String]
controls =
    [ "Neural network classify"
    , ""
    , "Left click to add a green dot"
    , "Right click to add a red dot"
    , "Click on an existing dot to remove it"
    , "Space bar to pause"
    , "Number to set hidden neurons"
    , "'l' to increase learning rate"
    , "';' to decrease learning rate"
    , "'r' to randomize weights"
    , "'c' to clear all dots"
    , "" ]
variables =
    [ "Error: "
    , "Hidden neurons: "
    , "Learning rate: "
    , "" ]

logStateInit :: State -> IO ()
logStateInit State {..} = do
    clearScreen
    hideCursor
    setCursorPosition 0 0
    mapM_ putStrLn $ controls ++ variables

logState :: State -> IO State
logState State {..} = do
    forM_ (zip3 [0..] variables values) $ \(i, var, val) -> do
        setCursorPosition (length controls + i) $ length var
        putStr val
    return State {..}
    where
    values =
        [ printf "%.8f" err
        , show hidden
        , printf "%.1f" rate
        , if running then "Running" else "Stopped" ]
