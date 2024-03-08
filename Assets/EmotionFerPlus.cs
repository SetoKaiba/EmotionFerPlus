using System.Linq;
using UnityEngine;
using Unity.Sentis;

public class EmotionFerplus : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;
    [SerializeField]
    Texture2D texture;
    IWorker m_Engine;
    TensorFloat m_Input;
    TensorFloat m_OutputTensor;

    void ReadbackCallback(bool completed)
    {
        m_OutputTensor.MakeReadable();
        var softmaxScores = Softmax(m_OutputTensor.ToReadOnlyArray());

        var maxIndex = System.Array.IndexOf(softmaxScores, softmaxScores.Max());
        var emotions = new string[] { "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt" };
        Debug.Log(emotions[maxIndex]);
    }
    
    // softmax method
    float[] Softmax(float[] values)
    {
        float max = values.Max();
        float scale = 0;
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Mathf.Exp(values[i] - max);
            scale += values[i];
        }
        for (int i = 0; i < values.Length; i++)
        {
            values[i] /= scale;
        }
        return values;
    }

    void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        
        m_Input = TextureConverter.ToTensor(texture, width: 64, height: 64);
        
        m_Engine.Execute(m_Input);

        m_OutputTensor = m_Engine.PeekOutput() as TensorFloat;
        
        m_OutputTensor.AsyncReadbackRequest(ReadbackCallback);
    }

    void OnDisable()
    {
        m_Engine.Dispose();
        m_Input.Dispose();
    }
}
